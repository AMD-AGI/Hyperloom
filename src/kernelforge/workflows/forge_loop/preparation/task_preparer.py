# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-loop task preparation ("self-healing") for forge-loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy, watchdog_timeout_sec, with_writable_sandbox
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.utils.git import ensure_commit_identity, git, git_apply, git_head
from kernelforge.config import Config
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.workflows.forge_loop.preparation.external_artifacts import (
    ExternalArtifactError,
    ExternalArtifactTransaction,
)
from kernelforge.config.resources import resource_path
from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.workflows.forge_loop.preparation.task_preflight import (
    DriverContract,
    PreflightResult,
    _preflight_async,
)
from kernelforge.workflows.forge_loop.preparation.task_preparer_audit import PrepareAudit
from kernelforge.workflows.forge_loop.preparation.task_preparer_prompts import (
    DRIVER_CONTRACT_SPEC,
    REFERENCE_DRIVER_TEMPLATE,
    RETRY_HEADING_NO_EDIT,
    _SYSTEM_PROMPT,
    PromptBrief,
    RetryNote,
    _build_prompt,
    _distributed_contract_note,
    _invocation_spec_note,
    _reference_note,
)

log = logging.getLogger(__name__)


# Bounded repair budget.
PREPARE_MAX_ATTEMPTS = int(os.environ.get("FORGE_PREPARE_MAX_ATTEMPTS", "3") or "3")
PREPARE_MAX_WALL_SEC = int(os.environ.get("FORGE_PREPARE_MAX_WALL", "3000") or "3000")
# Derived from the wall so it scales with it; a fixed constant falls below the wall/attempts ratio as soon as either
# grows.
PER_ATTEMPT_CAP_SEC = int(
    os.environ.get("FORGE_PREPARE_ATTEMPT_CAP") or max(1, PREPARE_MAX_WALL_SEC // max(1, PREPARE_MAX_ATTEMPTS))
)
# Smallest budget worth spending on a RETRY.
PREPARE_MIN_RETRY_SEC = int(os.environ.get("FORGE_PREPARE_MIN_RETRY", "350") or "350")
# Wall seconds reserved per attempt for the salvage preflight after a timeout.
_SALVAGE_RESERVE_SEC: float = float(os.environ.get("FORGE_SALVAGE_RESERVE", "120") or "120")

# Preflight bench is a quick format check, not a real measurement — keep it cheap.
PREFLIGHT_WARMUP = 3
PREFLIGHT_ITERS = 10

# The prep agent reads REAL reference example files (not just prompt text).
REFERENCE_SUBDIR = ".forge_task_reference"
INVOCATION_SPEC_FILENAME = "invocation_spec.json"
MAX_INVOCATION_SPEC_BYTES = 1024 * 1024
_INVOCATION_SPEC_NAME_RE = re.compile(r"^invocation_spec_[A-Za-z0-9._-]+\.json$")
_REFERENCE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "forge_experiments", ".git")


def preflight_task(
    *,
    driver: str,
    contract: DriverContract = DriverContract(),
    warmup: int = PREFLIGHT_WARMUP,
    iters: int = PREFLIGHT_ITERS,
    deadline_unix: float = 0.0,
) -> PreflightResult:
    """Synchronous deterministic check of a driver against the loop's contract."""

    return asyncio.run(_preflight_async(driver, contract, warmup, iters, deadline_unix=deadline_unix))


# Prepare — bounded LLM repair loop with snapshot/rollback


class ScaffoldRetirementError(RuntimeError):
    """The authoring-only reference bundle survived the retirement before a verdict."""


@dataclass
class PrepareResult:
    """Outcome of the preparation step (explicit success/failure contract)."""

    ok: bool
    attempts: int = 0
    wrote_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    rolled_back: bool = False
    final_preflight: PreflightResult | None = None
    message: str = ""
    audit_dir: str = ""


def _snapshot(paths: list[Path]) -> dict[Path, bytes | None]:
    """Record current bytes (or None if absent) for each path, for rollback.

    Taken before preparation touches anything, so an unreadable path aborts while the workspace is still the
    caller's: recording it as absent would make the rollback delete the file it was meant to protect.
    """
    return {p: (p.read_bytes() if p.is_file() else None) for p in paths}


def _restore(snapshot: dict[Path, bytes | None]) -> set[Path]:
    """Restore snapshotted paths — rewrite originals, delete ones that were absent — and report what it could not.

    Every entry is attempted: one path the filesystem will not give back must not cost the rest of the
    protected source its restoration. The paths returned are still carrying whatever the agent left there,
    so the caller owes the operator their names.
    """
    unrestored: set[Path] = set()
    for p, original in snapshot.items():
        try:
            if original is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(original)
        except OSError:
            unrestored.add(p)
    return unrestored


def _abs(workspace: Path, path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (workspace / p)


def _git(workspace: Path, *args: str) -> tuple[int, str]:
    """Run one git command, returning ``(exit code, stdout+stderr)``."""
    try:
        r = git(*args, cwd=workspace, check=False)
    except OSError as exc:
        return 128, str(exc)
    return r.returncode, (r.stdout + r.stderr)


def _git_diff_patch(workspace: Path, base_sha: str) -> str:
    """Capture the working tree's uncommitted tracked modifications vs HEAD."""
    if not base_sha:
        return ""
    return git("diff", "--binary", "HEAD", cwd=workspace).stdout


def _git_apply_patch(workspace: Path, patch: str) -> None:
    """Re-apply a patch captured by ``_git_diff_patch`` (no-op for an empty patch)."""
    if not patch.strip():
        return
    git_apply(patch, "--whitespace=nowarn", cwd=workspace)


def _git_untracked(workspace: Path) -> set[str]:
    """Set of untracked (and not-ignored) paths, relative to the workspace."""
    code, out = _git(workspace, "ls-files", "--others", "--exclude-standard")
    if code != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _stage_preparation_changes(workspace: Path, pre_untracked: set[str]) -> tuple[int, str]:
    """Stage tracked edits and only untracked files created by preparation."""
    code, output = _git(workspace, "add", "-u", "--", ".")
    if code != 0:
        return code, output
    created = sorted(
        path
        for path in (_git_untracked(workspace) - pre_untracked)
        if path != "forge_experiments" and not path.startswith("forge_experiments/")
    )
    if not created:
        return 0, output
    return _git(workspace, "add", "--", *created)


def _git_indexed(workspace: Path, path: Path) -> bool | None:
    """Whether ``path`` is in the workspace's index, i.e. will be committed."""
    try:
        relative = path.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    code, out = _git(workspace, "ls-files", "--cached", "--error-unmatch", "--", relative)
    if code == 0:
        return True
    # ``--error-unmatch`` exits 1 for a path the index does not hold; anything else is git failing, not git answering.
    return False if code == 1 else None


def _git_changed_since(workspace: Path, base_sha: str) -> list[str]:
    if not base_sha:
        return []
    code, out = _git(workspace, "diff", "--name-only", base_sha, "HEAD")
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _remove_new_untracked(workspace: Path, pre_untracked: set[str]) -> None:
    """Delete untracked files that appeared during prep (rollback of new files)."""
    for rel in _git_untracked(workspace) - pre_untracked:
        with contextlib.suppress(OSError):
            (workspace / rel).unlink()


def _safe_rmtree(path: Path | None) -> None:
    """Remove a tree best-effort; the caller checks whether it actually went."""
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def _safe_unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        if path.is_file():
            path.unlink()


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


def _ensure_agent_git_workspace(workspace: Path) -> None:
    """Create a private baseline commit when a backend requires a git cwd."""
    code, _ = _git(workspace, "rev-parse", "--show-toplevel")
    if code == 0:
        ensure_commit_identity(workspace)
        return
    code, output = _git(workspace, "init")
    if code != 0:
        raise RuntimeError(f"could not initialize preparation workspace: {output}")
    ensure_commit_identity(workspace)
    code, output = _git(workspace, "add", "-A")
    if code != 0:
        raise RuntimeError(f"could not stage preparation workspace: {output}")
    code, output = _git(
        workspace,
        "commit",
        "--allow-empty",
        "-m",
        "forge task preparation baseline",
    )
    if code != 0:
        raise RuntimeError(f"could not commit preparation workspace: {output}")


@dataclass(frozen=True)
class _AuthoringTurn:
    """One driver-authoring turn: where it runs, what it is told, its budget and what it may touch."""

    workspace: Path
    prompt: str
    timeout_sec: float
    target_files: Sequence[str] = ()
    protected_files: Sequence[str] = ()
    additional_dirs: list[str] | None = None
    allow_shell: bool = True


async def _run_prepare_agent(
    *,
    config: Config,
    turn: _AuthoringTurn,
    usage=None,
    progress_log: list[str] | None = None,
) -> str:
    """Run one sandboxed driver-authoring turn through the selected backend."""
    runtime = with_writable_sandbox(agent_runtime(config))
    backend = create_registered_backend(runtime)
    if backend.capabilities.requires_workspace_cwd:
        _ensure_agent_git_workspace(turn.workspace)
    protected_globs = list(dict.fromkeys(Path(path).name for path in (turn.protected_files or []) if path))
    spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=turn.prompt,
        role="task preparation",
        cwd=str(turn.workspace),
        writable=True,
        timeout_sec=max(1, int(turn.timeout_sec)),
        additional_directories=[directory for directory in (turn.additional_dirs or []) if directory],
        target_files=list(turn.target_files or []),
        # Deliberately no driver_script.
        protected_path_rules=FORGE_PROTECTED_PATHS,
        protected_globs=protected_globs,
        allow_dirty_targets=True,
        allow_untracked=True,
        # Preparation authors its own scaffolding before the agent starts: the reference bundle's harness/config files
        # and the durable invocation spec, all of which match protected_globs and none of which are targets.
        allow_dirty_baseline=True,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=turn.allow_shell,
            max_turns=50,
            permission_mode=os.environ.get(
                "FORGE_PERMISSION_MODE",
                "acceptEdits",
            ),
            bare=False,
        ),
        progress_log=progress_log,
    )
    result = await asyncio.wait_for(
        backend.run(spec, usage=usage),
        timeout=watchdog_timeout_sec(turn.timeout_sec),
    )
    return result.text.strip()


def _read_limited(path: Path, limit: int = 16000) -> str:
    try:
        return path.read_text(errors="replace")[:limit]
    except OSError:
        return ""


_COMPILE_ONLY_RE = re.compile(
    r"""print\s*\(\s*["']compile_only:\s*True["']\s*\)""",
)


def _is_compile_only_driver(text: str) -> bool:
    """True when the driver text is a compile-only autogen stub."""
    return _COMPILE_ONLY_RE.search(text) is not None


def _build_evidence(
    task: PreparationTask,
    *,
    workspace: Path,
    driver: str,
    preflight: PreflightResult | None,
) -> str:
    kernel, program_md = task.kernel, task.program_md
    target_functions, source_files = task.target_functions, task.source_files
    kernel_path = _abs(workspace, kernel)
    parts = [
        "## Task metadata",
        f"- workspace: `{workspace}`",
        f"- kernel (PROTECTED — DO NOT EDIT): `{kernel}`",
        f"- driver to create/fix (write here): `{driver}`",
        f"- target_functions: {', '.join(target_functions) or '(none given)'}",
    ]
    if source_files:
        parts.append("- other PROTECTED source files (DO NOT EDIT):")
        parts += [f"  - `{s}`" for s in source_files if s and _abs(workspace, s) != kernel_path]
    if program_md.strip():
        parts += ["", "## program.md (task guidance)", "```", program_md[:4000], "```"]
    parts += [
        "",
        "## Kernel source (the operator to measure — read its public entry point)",
        f"### `{kernel}`",
        "```python",
        _read_limited(kernel_path, 16000),
        "```",
    ]
    dpath = _abs(workspace, driver)
    if dpath.is_file():
        driver_text = _read_limited(dpath, 12000)
        if _is_compile_only_driver(driver_text):
            parts += [
                "",
                "## Current driver is a COMPILE-ONLY STUB — rewrite it completely",
                (
                    "The driver below only verifies that the kernel compiles with hipcc. "
                    "It has NO runtime measurement, NO correctness check, and NO timing "
                    "output. You MUST write a complete measurement driver from scratch — "
                    "do not adapt the compile-only boilerplate."
                ),
                f"### `{driver}`",
                "```python",
                driver_text,
                "```",
            ]
        else:
            parts += ["", "## Current (non-conforming) driver", f"### `{driver}`", "```python", driver_text, "```"]
    if preflight is not None and preflight.reasons:
        parts += ["", "## Why the current task fails the forge-loop contract", *[f"- {r}" for r in preflight.reasons]]
        for stage, tail in (preflight.diagnostics or {}).items():
            if tail and tail.strip():
                parts += ["", f"### What the {stage} stage actually printed (tail)", "```", tail.strip(), "```"]
    return "\n".join(parts)


def summarize_agent_progress(progress_log: list[str]) -> str:
    """Compact "what did it actually do" line from a streamed progress sink."""
    if not progress_log:
        return "no tool activity was recorded"
    non_streaming = any(entry.startswith("progress: not supported") for entry in progress_log)
    counts: dict[str, int] = {}
    for entry in progress_log:
        if entry.startswith("tool: "):
            name = entry[6:].split(" ", 1)[0]
            counts[name] = counts.get(name, 0) + 1
    parts = []
    if counts:
        parts.append(
            "tool calls: "
            + ", ".join(f"{name}x{count}" for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        )
    elif non_streaming:
        parts.append("backend does not support progress streaming")
    else:
        parts.append("no tool calls at all")
    tail = list(progress_log[-6:])
    if tail:
        parts.append("last steps:\n" + "\n".join(f"  {entry}" for entry in tail))
    return "; ".join(parts[:1]) + ("\n" + parts[1] if len(parts) > 1 else "")


@dataclass(frozen=True)
class PreparationTask:
    """The task preparation makes conform: its workspace, kernel, driver, guidance, sources and driver contract."""

    workspace_dir: str
    kernel: str
    driver: str
    program_md: str
    target_functions: list[str]
    source_files: list[str]
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    invocation_spec_file: str = ""
    expected_case_ids: Sequence[str] = ()
    read_only_files: Sequence[str] = ()
    nproc_per_node: int = 1


# Where each agent outcome snapshots the driver: after the agent, then after the preflight that judges it.
_ATTEMPT_SNAPSHOTS = {
    "completed": ("driver_after.py", "driver_at_preflight.py"),
    "timeout": ("driver_at_timeout.py", "driver_after_timeout_preflight.py"),
    "error": ("driver_at_exception.py", ""),
}


@dataclass(frozen=True)
class PreparationBudget:
    """How long preparation may run, and the outer per-kernel deadline that bounds it when set."""

    wall_sec: float = PREPARE_MAX_WALL_SEC
    deadline_unix: float = 0.0


async def prepare_task(
    *,
    config: Config,
    task: PreparationTask,
    budget: PreparationBudget = PreparationBudget(),
    preflight: PreflightResult | None = None,
    usage=None,
) -> PrepareResult:
    """Author/repair the driver so the task conforms; roll back on failure."""

    kernel, driver, source_files = task.kernel, task.driver, task.source_files
    invocation_spec_file, nproc_per_node = task.invocation_spec_file, task.nproc_per_node
    deadline_unix = budget.deadline_unix
    deadline_sec = _preparation_wall(budget.wall_sec, deadline_unix)
    workspace = Path(task.workspace_dir).resolve()
    driver_input_path = Path(os.path.abspath(os.path.expanduser(str(_abs(workspace, driver)))))
    driver_path = driver_input_path.resolve(strict=False)
    driver_external = not _driver_is_inside(workspace, driver_input_path, driver_path)

    audit_dirs = _preparation_audit_dir(config)
    audit_dir = audit_dirs.audit_dir
    audit_dir_str = str(audit_dir) if audit_dir is not None else ""

    external_transaction: ExternalArtifactTransaction | None = None
    agent_workspace = workspace
    additional_dirs = None
    # Kept apart from the bundle's own file list so an attempt whose re-materialization failed can drop that list and
    # keep these (see ``_open_scaffold``).
    reference_prefix = ""
    if driver_external:
        external_transaction, stage_error = _stage_external_driver(
            workspace,
            driver_input_path=driver_input_path,
            driver_path=driver_path,
            audit_dirs=audit_dirs,
            read_only_paths=[Path(path) for path in [*task.read_only_files, invocation_spec_file] if path],
        )
        if external_transaction is None:
            return PrepareResult(
                ok=False,
                attempts=0,
                rolled_back=True,
                message=stage_error,
                audit_dir=audit_dir_str,
            )
        driver_path = external_transaction.staged_driver_path
        agent_workspace = external_transaction.stage_root
        additional_dirs = [str(workspace)]
        reference_prefix = (
            "## Transactional external driver staging\n"
            f"The driver and every helper it imports MUST be written under the "
            f"isolated staging directory "
            f"`{driver_path.parent.resolve()}`. The kernel workspace is source evidence only; "
            "changes are published to the caller's artifact directory only after "
            "deterministic validation succeeds. Existing task metadata and invocation "
            "specifications are read-only.\n\n"
        )

    driver_access_dir = driver_path.parent.resolve()
    # Beside the driver in both cases, which is where the prompt says it is and
    # where the driver imports it from. Identical to the workspace root for a
    # driver that sits there; different only for one kept in a subdirectory,
    # where placing it at the root left the agent to find nothing and write its
    # own -- tripping the guard that protects harness files from being rewritten.
    harness_path = driver_access_dir / "graph_harness.py"
    # Unlike graph_harness, this one is not optional and is not retired when the
    # driver appears not to use it: a multi-rank driver that does not import it
    # is rejected by preflight, so its absence could only turn that verdict into
    # an ImportError.
    dist_harness_path = driver_access_dir / "dist_harness.py"

    audit = PrepareAudit(audit_dir, driver_path)

    if preflight is not None:
        audit.json("initial_preflight.json", asdict(preflight))

    # (1) Protect ONLY the source under optimization: kernel + declared source files.
    protected = {_abs(workspace, path) for path in (kernel, *source_files)}
    guard = _SourceGuard(workspace, protected)

    driver_rel = os.path.relpath(driver_path, agent_workspace)
    evidence = _build_evidence(task, workspace=workspace, driver=str(driver_path), preflight=preflight)

    # Give the agent REAL reference files to Read (copied into the workspace so they are inside its cwd).
    ref_dir = _materialize_reference(workspace)
    # The spec lives beside the driver, NOT in the reference bundle: the driver may read it at runtime, so it has to
    # outlive preparation and be committed alongside the driver it feeds.
    spec_path, canonical_spec = _place_invocation_spec(invocation_spec_file, driver_path.parent)
    invocation_note = _invocation_spec_note(spec_path, agent_workspace)
    contract = DriverContract(
        snr_threshold=task.snr_threshold,
        # A produced driver must actually time under a CUDA/HIP graph, not eagerly (nor silently fall back to eager).
        require_graph=True,
        require_profile=True,
        require_ranks=nproc_per_node,
        expected_case_ids=list(task.expected_case_ids),
    )
    canonical_harness, reference_prefix = _place_graph_harness(
        harness_path,
        ref_dir=ref_dir,
        driver_external=driver_external,
        reference_prefix=reference_prefix,
    )
    canonical_dist_harness = _place_dist_harness(dist_harness_path, nproc_per_node)
    reference_note = reference_prefix + _reference_note(ref_dir, agent_workspace)

    layout = _PreparationLayout(
        workspace=workspace,
        agent_workspace=agent_workspace,
        guard=guard,
        driver_external=driver_external,
        external_transaction=external_transaction,
        driver_path=driver_path,
        harness_path=harness_path,
        dist_harness_path=dist_harness_path,
        spec_path=spec_path,
        audit_dir_str=audit_dir_str,
    )
    scaffold = _PrepareScaffold(
        layout,
        reference=_ReferenceBundle(ref_dir, reference_prefix, reference_note),
        canonical=_CanonicalFiles(
            harness=canonical_harness,
            dist_harness=canonical_dist_harness,
            spec=canonical_spec,
        ),
    )
    backend_protected_files = _agent_protected_files(layout, protected, task)

    start = time.monotonic()
    preflight_deadline_unix = _preflight_deadline(deadline_sec, deadline_unix)
    attempts = 0
    retry_note = RetryNote()
    last_pf = preflight
    external_rollback_error = ""
    # An attempt that leaves the driver byte-identical is a distinct failure from one that edited it badly, and the
    # two need different guidance.
    edited_any_attempt = False
    starved_retry_sec = 0.0
    scaffold_error = ""
    try:
        while attempts < PREPARE_MAX_ATTEMPTS:
            budget = _attempt_budget(attempts, deadline_sec=deadline_sec, start=start)
            if not budget.may_start:
                starved_retry_sec = budget.starved_retry_sec
                break
            attempt_timeout = budget.timeout_sec
            attempts += 1
            # Each attempt authors against the reference bundle; the preceding attempt's verdict retired it (see
            # ``_PrepareScaffold.reset``).
            scaffold.open()
            prompt = _build_prompt(
                PromptBrief(evidence, driver_rel, invocation_note, _distributed_contract_note(nproc_per_node)),
                scaffold.reference_note,
                retry_note,
            )
            attempt_dir = f"attempt_{attempts:02d}"
            audit.text(f"{attempt_dir}/prompt.md", prompt)
            audit.text(f"{attempt_dir}/system_prompt.md", _SYSTEM_PROMPT)
            audit.driver(f"{attempt_dir}/driver_before.py")
            digest_before = audit.driver_digest()
            progress_log: list[str] = []
            agent_started = time.monotonic()

            status, exception_detail = "completed", {}
            try:
                agent_output = await _run_prepare_agent(
                    config=config,
                    turn=_AuthoringTurn(
                        workspace=agent_workspace,
                        prompt=prompt,
                        timeout_sec=attempt_timeout,
                        target_files=[
                            driver_path.as_posix(),
                            harness_path.as_posix(),
                        ],
                        protected_files=backend_protected_files,
                        additional_dirs=additional_dirs,
                        allow_shell=not driver_external,
                    ),
                    usage=usage,
                    progress_log=progress_log,
                )
                audit.text(f"{attempt_dir}/agent_output.txt", agent_output)
            except asyncio.TimeoutError:
                status = "timeout"
            except Exception as exc:  # isolation boundary: one task-preparation agent attempt
                log.warning("task preparation agent attempt %d failed: %s", attempts, exc, exc_info=True)
                status = "error"
                exception_detail = {"exception_type": type(exc).__name__, "exception": str(exc)}
                retry_note = RetryNote(f"Agent invocation error: {type(exc).__name__}: {exc}")
            audit.text(f"{attempt_dir}/agent_progress.txt", "\n".join(progress_log))
            agent_snapshot, judged_snapshot = _ATTEMPT_SNAPSHOTS[status]
            audit.driver(f"{attempt_dir}/{agent_snapshot}")
            elapsed_s = round(time.monotonic() - agent_started, 3)
            driver_edited = audit.driver_edited(digest_before)
            edited_any_attempt = edited_any_attempt or driver_edited
            audit.json(
                f"{attempt_dir}/agent_event.json",
                {
                    "status": status,
                    **exception_detail,
                    "elapsed_s": elapsed_s,
                    "budget_s": round(attempt_timeout, 3),
                    "driver_edited": driver_edited,
                },
            )

            # (1) Source protection: whatever the agent did, restore source (and the provided harness) to pristine
            # before we judge the driver.
            scaffold.reset()
            if status == "error":
                continue

            # A timed-out agent may have completed a valid driver before getting stuck on self-verification.
            last_pf = await _preflight_async(
                driver_path.as_posix(),
                contract,
                PREFLIGHT_WARMUP,
                PREFLIGHT_ITERS,
                deadline_unix=preflight_deadline_unix,
            )
            audit.driver(f"{attempt_dir}/{judged_snapshot}")
            audit.json(f"{attempt_dir}/preflight.json", asdict(last_pf))
            if last_pf.ok:
                return _finish_prepared_task(last_pf, attempts, layout, provided_harness=canonical_harness is not None)
            if status == "timeout":
                retry_note = _timeout_retry_prompt(
                    last_pf,
                    driver_rel=driver_rel,
                    driver_edited=driver_edited,
                    elapsed_s=elapsed_s,
                    progress_log=progress_log,
                )
            else:
                retry_note = _preflight_retry_prompt(last_pf, driver_rel=driver_rel, driver_edited=driver_edited)
    except ScaffoldRetirementError as exc:
        scaffold_error = str(exc)
    finally:
        # Never leave reference or external staging bundles behind.
        _safe_rmtree(ref_dir)
        if external_transaction is not None:
            external_rollback_error = _close_external_transaction(external_transaction, guard)

    # (3) Failure: roll the workspace back to its exact pre-prep state (tracked files reset to HEAD, caller's
    # uncommitted mods re-applied, prep-created untracked removed, protected source restored).
    guard.rollback()
    return PrepareResult(
        ok=False,
        attempts=attempts,
        wrote_files=[],
        created_files=[],
        rolled_back=not external_rollback_error and not guard.unrestored,
        final_preflight=last_pf,
        message=_prepare_failure_message(
            guard,
            _PrepareFailure(
                scaffold_error=scaffold_error,
                attempts=attempts,
                starved_retry_sec=starved_retry_sec,
                edited_any_attempt=edited_any_attempt,
                external_rollback_error=external_rollback_error,
            ),
        ),
        audit_dir=audit_dir_str,
    )


def prepare_task_sync(**kwargs) -> PrepareResult:
    """Synchronous wrapper for CLI code."""

    return asyncio.run(prepare_task(**kwargs))


def _preparation_wall(deadline_sec: float, deadline_unix: float) -> float:
    """How long preparation may run, bounded by the outer per-kernel deadline when there is one."""
    if deadline_unix <= 0:
        return deadline_sec
    return min(deadline_sec, max(0.0, deadline_unix - time.time()))


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


class _SourceGuard:
    """The source under optimization, and putting the workspace back exactly as preparation found it.

    The rollback anchors are a byte-snapshot of the protected source
    (guaranteed restore even if untracked) plus the git state, so ANY other
    file the agent creates or modifies can be undone on failure without knowing
    it in advance.
    """

    def __init__(self, workspace: Path, protected: set[Path]) -> None:
        self._workspace = workspace
        self._snapshot = _snapshot(list(protected))
        self.base_sha = git_head(workspace)
        self.pre_untracked = _git_untracked(workspace)
        # The caller's pre-prep uncommitted tracked modifications, captured so a failure rollback restores them
        # instead of resetting the whole tree to HEAD.
        self._pre_diff = _git_diff_patch(workspace, self.base_sha)
        # Protected paths the rollback could not put back. Preparation cannot be declared successful while the
        # caller's own source still carries the prep agent's edits, so this outranks any verdict on the driver.
        self.unrestored: set[Path] = set()

    def restore_protected(self) -> None:
        """Put the protected source bytes back, which covers untracked source too."""
        # Every round attempts the whole snapshot, so the latest round is the complete answer and a path that came
        # back on a retry is no longer unrestored.
        self.unrestored.clear()
        self.unrestored.update(_restore(self._snapshot))

    def rollback(self) -> None:
        """Undo everything the agent did while preserving the caller's pre-prep state.

        Tracked files are reset to HEAD, only prep-created untracked files are
        dropped, the caller's original uncommitted tracked modifications are
        re-applied, and the protected source bytes are then restored
        authoritatively.
        """
        if self.base_sha:
            _git(self._workspace, "reset", "-q")
            _git(self._workspace, "checkout", "--", ".")
            _remove_new_untracked(self._workspace, self.pre_untracked)
            _git_apply_patch(self._workspace, self._pre_diff)
        self.restore_protected()

    def unrestored_message(self) -> str:
        """Why preparation cannot be declared successful, whatever the driver did."""
        named = ", ".join(path.as_posix() for path in sorted(self.unrestored))
        return (
            f"could not restore the protected source after preparation: {named}; the workspace still holds "
            "the prep agent's edits to it, so the task was not prepared"
        )


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


@dataclass(frozen=True)
class _PreparationLayout:
    """Where one preparation runs: the workspace, the agent's tree, the guard and the files it authors."""

    workspace: Path
    agent_workspace: Path
    guard: "_SourceGuard"
    driver_external: bool
    external_transaction: ExternalArtifactTransaction | None
    driver_path: Path
    harness_path: Path
    dist_harness_path: Path
    spec_path: Path | None
    audit_dir_str: str


def _agent_protected_files(layout: _PreparationLayout, protected: set[Path], task: PreparationTask) -> list[str]:
    """What the authoring agent may not write: the source under optimization and the inputs preparation ships."""
    return [
        *(path.as_posix() for path in protected),
        *task.read_only_files,
        *([layout.spec_path.as_posix()] if layout.spec_path is not None else []),
        # Protected rather than merely discouraged: the distributed guarantees
        # are only guarantees while this file is the one KernelForge shipped.
        *([layout.dist_harness_path.as_posix()] if task.nproc_per_node > 1 else []),
    ]


@dataclass(frozen=True)
class _ReferenceBundle:
    """The authoring-only reference bundle and the prompt note that points the agent at it."""

    ref_dir: Path | None
    prefix: str
    note: str


@dataclass(frozen=True)
class _CanonicalFiles:
    """The shipped contents restored after every attempt: graph harness, distributed harness and spec."""

    harness: str | None
    dist_harness: str | None
    spec: str


class _PrepareScaffold:
    """The authoring-only reference bundle, and the state the driver is judged in.

    Each attempt authors against the bundle and the attempt's verdict retires
    it, so putting it back and taking it away are two halves of one job. What
    is left behind after a reset is exactly what the loop's baseline will run.
    """

    def __init__(self, layout: _PreparationLayout, *, reference: _ReferenceBundle, canonical: _CanonicalFiles) -> None:
        self._workspace = layout.workspace
        self._ref_dir = reference.ref_dir
        self._reference_prefix = reference.prefix
        self.reference_note = reference.note
        self._agent_workspace = layout.agent_workspace
        self._guard = layout.guard
        self._external_transaction = layout.external_transaction
        self._harness_path = layout.harness_path
        self._canonical_harness = canonical.harness
        self._dist_harness_path = layout.dist_harness_path
        self._canonical_dist_harness = canonical.dist_harness
        self._spec_path = layout.spec_path
        self._canonical_spec = canonical.spec

    def open(self) -> None:
        """Put the authoring-only reference bundle back for the next attempt."""
        if self._ref_dir is None or self._ref_dir.is_dir():
            return
        materialized = _materialize_reference(self._workspace)
        if materialized is None:
            # The note enumerates the contract and the reference drivers by path, in a prompt that also tells the
            # agent not to rely on memory, so keeping it would send this attempt to Read files that are gone.
            log.warning(
                "could not re-materialize the authoring reference bundle at %s; "
                "this attempt's prompt drops its file list",
                self._ref_dir,
            )
        self.reference_note = self._reference_prefix + _reference_note(materialized, self._agent_workspace)

    def reset(self) -> None:
        """Put the workspace into the exact state the driver will be judged in."""
        if self._external_transaction is not None:
            self._guard.rollback()
            self._external_transaction.restore_passthroughs()
        else:
            self._guard.restore_protected()
        _safe_rmtree(self._ref_dir)
        if self._ref_dir is not None and self._ref_dir.exists():
            # A partial removal leaves both halves of the invariant broken at once and neither is visible later:
            # preflight judges the driver against scaffolding the prep commit then deletes, and `git add -A` carries
            # what survived into the pristine commit.
            raise ScaffoldRetirementError(
                f"could not retire the authoring reference bundle at {self._ref_dir}; the "
                "driver would be validated against scaffolding that preparation "
                "then deletes, and the surviving files would enter the pristine "
                "commit"
            )
        if self._canonical_harness is not None:
            self._harness_path.write_text(self._canonical_harness)
        if self._canonical_dist_harness is not None:
            self._dist_harness_path.write_text(self._canonical_dist_harness, encoding="utf-8")
        if self._spec_path is not None and self._canonical_spec:
            self._spec_path.parent.mkdir(parents=True, exist_ok=True)
            self._spec_path.write_text(self._canonical_spec, encoding="utf-8")


def _finish_prepared_task(
    pf: PreflightResult,
    attempt_count: int,
    layout: _PreparationLayout,
    *,
    provided_harness: bool,
) -> PrepareResult:
    """Sign off a driver that conforms, by whichever route this task is delivered."""
    guard, external_transaction, audit_dir_str = layout.guard, layout.external_transaction, layout.audit_dir_str
    driver_external = layout.driver_external
    if driver_external:
        # Publish the complete validated driver/helper change set from the isolated staging tree.
        guard.rollback()
        assert external_transaction is not None
        external_transaction.restore_passthroughs()
    elif provided_harness:
        # Drop an unused provided harness so it does not become persistent scaffolding when the driver does not
        # import it.
        try:
            uses_harness = "graph_harness" in layout.driver_path.read_text()
        except OSError:
            uses_harness = True
        if not uses_harness:
            _safe_unlink(layout.harness_path)
    if guard.unrestored:
        # Signing off here would carry the agent's edit of the source under optimization into the pristine
        # commit, and every measurement after it would judge a kernel the caller never wrote.
        return PrepareResult(
            ok=False,
            attempts=attempt_count,
            wrote_files=[],
            created_files=[],
            rolled_back=False,
            final_preflight=pf,
            message=guard.unrestored_message(),
            audit_dir=audit_dir_str,
        )
    if driver_external:
        return _publish_external_preparation(external_transaction, pf, attempt_count, audit_dir_str)
    return _commit_prepared_task(layout, pf=pf, attempt_count=attempt_count)


def _preflight_deadline(deadline_sec: float, deadline_unix: float) -> float:
    """When the in-loop preflight must stop.

    It respects the preparation wall as well as the outer per-kernel deadline,
    so a preflight cannot outlive the preparation that asked for it.
    """
    wall = time.time() + deadline_sec
    return min(wall, deadline_unix) if deadline_unix > 0 else wall


@dataclass(frozen=True)
class _AttemptBudget:
    """Whether the next preparation attempt may start, and what it may spend.

    A zero spend is not a refusal, which is why the two are separate fields: the
    attempt still runs, ``_run_prepare_agent`` clamps the timeout to a second,
    and the salvage preflight that follows the timeout is a real chance to pass.
    Only running out of wall clock, or a retry that would get less than
    ``PREPARE_MIN_RETRY_SEC``, stops the loop.
    """

    may_start: bool
    timeout_sec: float = 0.0
    starved_retry_sec: float = 0.0


def _attempt_budget(attempts: int, *, deadline_sec: float, start: float) -> _AttemptBudget:
    """What the next attempt may spend, and whether it may start at all.

    The last attempt may spend everything but the salvage reserve; earlier ones
    are capped so a later one still gets a turn.
    """
    remaining = deadline_sec - (time.monotonic() - start)
    if remaining <= 10:
        return _AttemptBudget(may_start=False)
    if attempts and remaining < PREPARE_MIN_RETRY_SEC:
        return _AttemptBudget(may_start=False, starved_retry_sec=remaining)
    spendable = max(0.0, remaining - _SALVAGE_RESERVE_SEC)
    is_last_attempt = attempts + 1 >= PREPARE_MAX_ATTEMPTS
    timeout = spendable if is_last_attempt else min(spendable, float(PER_ATTEMPT_CAP_SEC))
    return _AttemptBudget(may_start=True, timeout_sec=timeout)


def _jit_timeout_hint(pf: PreflightResult) -> str:
    """Say when every preflight failure was a timeout, which usually means slow JIT rather than a broken driver."""
    if not pf.all_failures_are_timeouts:
        return ""
    return (
        "\nNOTE: Every failure above is a TIMEOUT, not a crash. "
        "This usually means the driver is structurally correct "
        "but the first execution triggered slow JIT compilation "
        "(CK/aiter kernels can take 44s+ per module). Do NOT "
        "rewrite the driver from scratch — verify it is "
        "structurally correct and resubmit; the next preflight "
        "run benefits from a warm JIT cache.\n"
    )


def _timeout_retry_prompt(
    pf: PreflightResult,
    *,
    driver_rel: str,
    driver_edited: bool,
    elapsed_s: float,
    progress_log: list[str],
) -> RetryNote:
    """What the next attempt is told after this one ran out of time."""
    if driver_edited:
        return RetryNote(
            "Agent timed out, then deterministic preflight failed:\n" + pf.detail_report() + _jit_timeout_hint(pf)
        )
    return RetryNote(
        f"Your previous attempt ran for {elapsed_s:.0f}s and timed out "
        f"having made NO edit at all to `{driver_rel}` — the file is "
        "byte-identical to before. Reading and planning is not "
        "progress here. Open the driver and WRITE the fix as your "
        "first substantive action, then verify; if you cannot finish "
        "the whole contract in the time you have, still leave the "
        "best driver you can on disk rather than nothing.\n"
        f"What you spent that time on — {summarize_agent_progress(progress_log)}\n"
        "The deterministic check on that unchanged driver reported:\n" + pf.detail_report(),
        RETRY_HEADING_NO_EDIT,
    )


def _preflight_retry_prompt(pf: PreflightResult, *, driver_rel: str, driver_edited: bool) -> RetryNote:
    """What the next attempt is told after this one finished but left a driver that does not conform."""
    if driver_edited:
        return RetryNote("Deterministic preflight after your edit:\n" + pf.detail_report() + _jit_timeout_hint(pf))
    return RetryNote(
        f"You finished without editing `{driver_rel}` at all. The "
        "deterministic check therefore ran the SAME driver again:\n" + pf.detail_report(),
        RETRY_HEADING_NO_EDIT,
    )


def _close_external_transaction(external_transaction, guard: "_SourceGuard") -> str:
    """Roll back and close the staging transaction; return what it could not clean up."""
    error = ""
    if not external_transaction.published:
        try:
            external_transaction.rollback()
        except ExternalArtifactError as exc:
            error = str(exc)
    if not external_transaction.published:
        guard.rollback()
    try:
        external_transaction.close()
    except OSError as exc:
        if not error:
            error = f"staging cleanup failed: {exc}"
    return error


@dataclass(frozen=True)
class _PrepareFailure:
    """How a preparation ended without a conforming driver."""

    scaffold_error: str
    attempts: int
    starved_retry_sec: float
    edited_any_attempt: bool
    external_rollback_error: str


def _prepare_failure_message(guard: "_SourceGuard", failure: _PrepareFailure) -> str:
    """Why preparation failed, led by whichever fact the operator has to deal with first."""
    if guard.unrestored:
        # The operator has a file on disk that is not theirs any more, which they have to deal with before any
        # reason the driver was refused.
        log.error("%s", guard.unrestored_message())
        return guard.unrestored_message()
    scaffold_error, attempts, starved_retry_sec = failure.scaffold_error, failure.attempts, failure.starved_retry_sec
    if scaffold_error:
        # The preflight reasons describe a driver judged in a state that was never valid, so quoting them first
        # would send the operator after the driver.
        log.error("%s", scaffold_error)
        return scaffold_error
    if starved_retry_sec:
        reason = (
            f"prep wall exhausted after {attempts} attempt(s); the remaining "
            f"{starved_retry_sec:.0f}s is below the {PREPARE_MIN_RETRY_SEC}s "
            "minimum retry budget, so no further attempt was started — raise "
            "the per-kernel deadline to give preparation more room"
        )
    elif failure.edited_any_attempt or not attempts:
        reason = "could not produce a conforming driver within the budget"
    else:
        reason = (
            f"prep agent never edited the driver in {attempts} attempt(s); "
            "the driver is unchanged, so the preflight reasons below describe "
            "the ORIGINAL driver, not a failed repair"
        )
    if failure.external_rollback_error:
        reason += f"; external artifact rollback failed: {failure.external_rollback_error}"
    return reason


def _commit_prepared_task(layout: _PreparationLayout, *, pf, attempt_count: int) -> PrepareResult:
    """Make the prepared driver, and everything it reads at runtime, part of pristine.

    IterationLoop captures its base SHA from this commit, so anything left out of it is not there for the measurements
    that follow. Nothing staged is nothing to commit: a driver git ignores, as the controller stages one, runs in place.
    """
    workspace, guard, spec_path, audit_dir_str = layout.workspace, layout.guard, layout.spec_path, layout.audit_dir_str
    stage_code, stage_out = _stage_preparation_changes(workspace, guard.pre_untracked)
    if stage_code != 0:
        guard.rollback()
        return PrepareResult(
            ok=False,
            attempts=attempt_count,
            wrote_files=[],
            created_files=[],
            rolled_back=True,
            final_preflight=pf,
            message=f"could not stage prepared task files: {stage_out.strip()[-200:]}",
            audit_dir=audit_dir_str,
        )
    spec_indexed = None if spec_path is None else _git_indexed(workspace, spec_path)
    if spec_path is not None and spec_indexed is not True:
        guard.rollback()
        return PrepareResult(
            ok=False,
            attempts=attempt_count,
            wrote_files=[],
            created_files=[],
            rolled_back=True,
            final_preflight=pf,
            message=(
                "prepared a conforming driver but its invocation "
                f"specification ({spec_path}) cannot be committed alongside "
                "it, so the driver's runtime input would not be durable — "
                + (
                    "check the workspace's git ignore rules"
                    if spec_indexed is False
                    else "git could not be asked whether it is staged, so the "
                    "ignore rules are not necessarily the reason"
                )
            ),
            audit_dir=audit_dir_str,
        )
    # The agent path that names an identity runs only when the backend needs
    # a git cwd, and a commit with none to auto-detect cannot land.
    ensure_commit_identity(workspace)
    _, commit_out = _git(workspace, "commit", "-m", "forge prepass: prepared measurement driver")
    if git_head(workspace) == guard.base_sha and _git(workspace, "diff-index", "--quiet", "--cached", "HEAD")[0]:
        guard.rollback()
        return PrepareResult(
            ok=False,
            attempts=attempt_count,
            wrote_files=[],
            created_files=[],
            rolled_back=True,
            final_preflight=pf,
            message=(f"prepared a conforming driver but the git commit did not land: {commit_out.strip()[-200:]}"),
            audit_dir=audit_dir_str,
        )
    return PrepareResult(
        ok=True,
        attempts=attempt_count,
        wrote_files=_git_changed_since(workspace, guard.base_sha),
        created_files=[],
        rolled_back=False,
        final_preflight=pf,
        message="task prepared",
        audit_dir=audit_dir_str,
    )


def _publish_external_preparation(external_transaction, pf, attempt_count: int, audit_dir_str: str) -> PrepareResult:
    """Publish the validated driver and helper change set out of the isolated staging tree."""
    try:
        changes = external_transaction.publish()
    except ExternalArtifactError as exc:
        rollback_error = ""
        try:
            external_transaction.rollback()
        except ExternalArtifactError as rollback_exc:
            rollback_error = f"; rollback failed: {rollback_exc}"
        return PrepareResult(
            ok=False,
            attempts=attempt_count,
            wrote_files=[],
            created_files=[],
            rolled_back=not rollback_error,
            final_preflight=pf,
            message=f"could not publish external driver artifacts: {exc}{rollback_error}",
            audit_dir=audit_dir_str,
        )
    return PrepareResult(
        ok=True,
        attempts=attempt_count,
        wrote_files=list(changes.wrote_files),
        created_files=list(changes.created_files),
        rolled_back=False,
        final_preflight=pf,
        message="external task prepared",
        audit_dir=audit_dir_str,
    )
