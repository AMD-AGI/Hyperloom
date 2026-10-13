# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-loop task preparation ("self-healing") for forge-loop."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy, watchdog_timeout_sec, with_writable_sandbox
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.llm.git import ensure_commit_identity, git
from kernelforge.config import Config
from kernelforge.loop.external_artifacts import ExternalArtifactError, ExternalArtifactTransaction
from kernelforge.resources import resource_path
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.loop.task_preflight import PreflightResult, _preflight_async
from kernelforge.loop.task_preparer_prompts import (
    DRIVER_CONTRACT_SPEC,
    REFERENCE_DRIVER_TEMPLATE,
    RETRY_HEADING_DEFAULT,
    RETRY_HEADING_NO_EDIT,
    _SYSTEM_PROMPT,
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
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    warmup: int = PREFLIGHT_WARMUP,
    iters: int = PREFLIGHT_ITERS,
    require_graph: bool = False,
    require_profile: bool = False,
    require_ranks: int = 1,
    deadline_unix: float = 0.0,
    expected_case_ids: list[str] | None = None,
) -> PreflightResult:
    """Synchronous deterministic check of a driver against the loop's contract."""

    return asyncio.run(
        _preflight_async(
            driver,
            snr_threshold,
            warmup,
            iters,
            require_graph,
            require_profile,
            require_ranks,
            deadline_unix,
            expected_case_ids,
        )
    )


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


def _git_head(workspace: Path) -> str:
    code, out = _git(workspace, "rev-parse", "HEAD")
    return out.strip() if code == 0 else ""


def _git_diff_patch(workspace: Path, base_sha: str) -> str:
    """Capture the working tree's uncommitted tracked modifications vs HEAD."""
    if not base_sha:
        return ""
    return git("diff", "--binary", "HEAD", cwd=workspace).stdout


def _git_apply_patch(workspace: Path, patch: str) -> None:
    """Re-apply a patch captured by ``_git_diff_patch`` (no-op for an empty patch)."""
    if not patch.strip():
        return
    git("apply", "--whitespace=nowarn", cwd=workspace, input=patch)


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


async def _run_prepare_agent(
    *,
    config: Config,
    workspace: Path,
    system_prompt: str,
    prompt: str,
    timeout_sec: float,
    additional_dirs: list[str] | None = None,
    allow_shell: bool = True,
    target_files: list[str] | None = None,
    protected_files: list[str] | None = None,
    usage=None,
    progress_log: list[str] | None = None,
) -> str:
    """Run one sandboxed driver-authoring turn through the selected backend."""
    runtime = with_writable_sandbox(config.agent_runtime())
    backend = create_registered_backend(runtime)
    if backend.capabilities.requires_workspace_cwd:
        _ensure_agent_git_workspace(workspace)
    protected_globs = list(dict.fromkeys(Path(path).name for path in (protected_files or []) if path))
    spec = AgentRunSpec(
        system_prompt=system_prompt,
        user_prompt=prompt,
        role="task preparation",
        cwd=str(workspace),
        writable=True,
        timeout_sec=max(1, int(timeout_sec)),
        additional_directories=[directory for directory in (additional_dirs or []) if directory],
        target_files=list(target_files or []),
        # Deliberately no driver_script.
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
            shell=allow_shell,
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
        timeout=watchdog_timeout_sec(timeout_sec),
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
    *,
    workspace: Path,
    kernel: str,
    driver: str,
    program_md: str,
    target_functions: list[str],
    source_files: list[str],
    preflight: PreflightResult | None,
) -> str:
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
                "The driver below only verifies that the kernel compiles with hipcc. "
                "It has NO runtime measurement, NO correctness check, and NO timing "
                "output. You MUST write a complete measurement driver from scratch — "
                "do not adapt the compile-only boilerplate.",
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


async def prepare_task(
    *,
    config: Config,
    workspace_dir: str,
    kernel: str,
    driver: str,
    program_md: str,
    target_functions: list[str],
    source_files: list[str],
    kernel_backend: str = "",
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    preflight: PreflightResult | None = None,
    deadline_sec: float = PREPARE_MAX_WALL_SEC,
    deadline_unix: float = 0.0,
    invocation_spec_file: str = "",
    expected_case_ids: list[str] | None = None,
    read_only_files: list[str] | None = None,
    nproc_per_node: int = 1,
    usage=None,
) -> PrepareResult:
    """Author/repair the driver so the task conforms; roll back on failure."""

    if deadline_unix > 0:
        deadline_sec = min(
            deadline_sec,
            max(0.0, deadline_unix - time.time()),
        )
    workspace = Path(workspace_dir).resolve()
    driver_input_path = Path(os.path.abspath(os.path.expanduser(str(_abs(workspace, driver)))))
    driver_path = driver_input_path.resolve(strict=False)
    try:
        driver_input_path.relative_to(workspace)
        lexical_driver_internal = True
    except ValueError:
        lexical_driver_internal = False
    try:
        driver_path.relative_to(workspace)
        resolved_driver_internal = True
    except ValueError:
        resolved_driver_internal = False
    driver_external = not (lexical_driver_internal and resolved_driver_internal)

    experiments_dir: Path | None = None
    audit_dir: Path | None = None
    try:
        experiments_dir = Path(config.experiments_dir)
        audit_dir = experiments_dir / "task_preparation"
        audit_dir.mkdir(parents=True, exist_ok=True)
    except (AttributeError, OSError, TypeError, ValueError):
        audit_dir = None
    audit_dir_str = str(audit_dir) if audit_dir is not None else ""

    external_transaction: ExternalArtifactTransaction | None = None
    agent_workspace = workspace
    if driver_external:
        external_exclusions = [workspace]
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
        protected_external_inputs = [Path(path) for path in [*(read_only_files or []), invocation_spec_file] if path]
        try:
            external_transaction = ExternalArtifactTransaction(
                driver_path=driver_input_path,
                excluded_paths=external_exclusions,
                passthrough_paths=[workspace],
                read_only_paths=protected_external_inputs,
            )
        except ExternalArtifactError as exc:
            return PrepareResult(
                ok=False,
                attempts=0,
                rolled_back=True,
                message=f"could not stage external driver artifacts: {exc}",
                audit_dir=audit_dir_str,
            )
        driver_path = external_transaction.staged_driver_path
        agent_workspace = external_transaction.stage_root

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

    def _audit_text(relative: str, text: str) -> None:
        if audit_dir is None:
            return
        try:
            path = audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            # Audit artifacts are best-effort and never affect driver validity.
            pass

    def _audit_json(relative: str, payload: dict) -> None:
        try:
            _audit_text(
                relative,
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            )
        except (TypeError, ValueError):
            # Non-serializable audit metadata must not block task preparation.
            pass

    def _driver_digest() -> str:
        try:
            return hashlib.sha256(driver_path.read_bytes()).hexdigest()
        except OSError:
            return ""

    def _detect_driver_edited(digest_before: str) -> bool:
        return _driver_digest() != digest_before

    def _audit_driver(relative: str) -> None:
        if audit_dir is None or not driver_path.is_file():
            return
        try:
            path = audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(driver_path, path)
            # copy2 carries the SOURCE mtime across, so every snapshot in the audit trail claimed the driver's own
            # mtime rather than when it was captured — reconstructing a timeline from this directory (the obvious
            # thing to do when a prep fails) then yields wildly wrong durations.
            os.utime(path, None)
        except OSError:
            # A missing audit copy is non-fatal; the staged driver remains the authoritative preparation artifact.
            pass

    if preflight is not None:
        _audit_json("initial_preflight.json", asdict(preflight))

    # (1) Protect ONLY the source under optimization: kernel + declared source files.
    protected = {_abs(workspace, kernel)}
    for s in source_files:
        if s:
            protected.add(_abs(workspace, s))

    # Rollback anchors: a byte-snapshot of the protected source (guaranteed restore even if untracked) plus the git
    # state, so ANY other file the agent creates/modifies can be undone on failure without knowing it in advance.
    src_snapshot = _snapshot(list(protected))
    prep_base_sha = _git_head(workspace)
    pre_untracked = _git_untracked(workspace)
    # The caller's pre-prep uncommitted tracked modifications, captured so a failure rollback restores them instead of
    # resetting the whole tree to HEAD.
    pre_diff = _git_diff_patch(workspace, prep_base_sha)

    # Protected paths the rollback could not put back. Preparation cannot be declared successful while the
    # caller's own source still carries the prep agent's edits, so this outranks any verdict on the driver.
    unrestored_sources: set[Path] = set()

    def _restore_protected() -> None:
        # Every round attempts the whole snapshot, so the latest round is the complete answer and a path that came
        # back on a retry is no longer unrestored.
        unrestored_sources.clear()
        unrestored_sources.update(_restore(src_snapshot))

    def _unrestored_message() -> str:
        named = ", ".join(p.as_posix() for p in sorted(unrestored_sources))
        return (
            f"could not restore the protected source after preparation: {named}; the workspace still holds "
            "the prep agent's edits to it, so the task was not prepared"
        )

    def _restore_kernel_workspace() -> None:
        # Undo everything the agent did while preserving the caller's pre-prep state: reset tracked files to HEAD,
        # drop only prep-created untracked files, re-apply the caller's original uncommitted tracked modifications,
        # then authoritatively restore the protected source bytes (covers untracked source too).
        if prep_base_sha:
            _git(workspace, "reset", "-q")
            _git(workspace, "checkout", "--", ".")
            _remove_new_untracked(workspace, pre_untracked)
            _git_apply_patch(workspace, pre_diff)
        _restore_protected()

    def _rollback() -> None:
        _restore_kernel_workspace()

    driver_rel = os.path.relpath(driver_path, agent_workspace)
    evidence = _build_evidence(
        workspace=workspace,
        kernel=kernel,
        driver=str(driver_path),
        program_md=program_md,
        target_functions=target_functions,
        source_files=source_files,
        preflight=preflight,
    )

    # Give the agent REAL reference files to Read (copied into the workspace so they are inside its cwd).
    ref_dir = _materialize_reference(workspace)
    # Kept apart from the bundle's own file list so an attempt whose re-materialization failed can drop that list and
    # keep these (see ``_open_scaffold``).
    reference_prefix = ""
    if driver_external:
        reference_prefix = (
            "## Transactional external driver staging\n"
            f"The driver and every helper it imports MUST be written under the "
            f"isolated staging directory "
            f"`{driver_access_dir}`. The kernel workspace is source evidence only; "
            "changes are published to the caller's artifact directory only after "
            "deterministic validation succeeds. Existing task metadata and invocation "
            "specifications are read-only.\n\n"
        )
    # The spec lives beside the driver, NOT in the reference bundle: the driver may read it at runtime, so it has to
    # outlive preparation and be committed alongside the driver it feeds.
    spec_path, canonical_spec = _materialize_invocation_spec(
        invocation_spec_file,
        driver_path.parent,
    )
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
    invocation_note = _invocation_spec_note(spec_path, agent_workspace)
    expected_case_ids = list(expected_case_ids or [])
    backend_protected_files = [
        *(path.as_posix() for path in protected),
        *(read_only_files or []),
        *([spec_path.as_posix()] if spec_path is not None else []),
        # Protected rather than merely discouraged: the distributed guarantees
        # are only guarantees while this file is the one KernelForge shipped.
        *([dist_harness_path.as_posix()] if nproc_per_node > 1 else []),
    ]

    # (2) Pre-place a correct, capture-guarded graph_harness.py so the agent can import a known-good cuda_graph_bench
    # (accepting dirty/verify) instead of writing its own — a self-written harness can silently mismatch its driver
    # calls and degrade graph timing to eager.
    canonical_harness = None if harness_path.is_file() else _find_reference_harness(ref_dir)
    provided_harness = canonical_harness is not None
    if provided_harness:
        harness_path.write_text(canonical_harness)
        harness_display = str(harness_path) if driver_external else "./graph_harness.py"
        reference_prefix = (
            "## Graph timing harness (already available beside the driver)\n"
            f"`{harness_display}` is a correct, capture-guarded harness. Import it —\n"
            "`from graph_harness import cuda_graph_bench` (it accepts optional\n"
            "`dirty`/`verify`). Do NOT rewrite it. Only implement custom timing in the\n"
            "driver if this operator genuinely cannot be captured into a static-input\n"
            "graph.\n\n" + reference_prefix
        )

    # (3) A multi-rank task measures inside dist_harness or it is not measured,
    # so the module is placed before the agent is asked for anything.
    canonical_dist_harness = _dist_harness_text() if nproc_per_node > 1 else None
    if canonical_dist_harness is not None:
        dist_harness_path.write_text(canonical_dist_harness, encoding="utf-8")
    elif nproc_per_node > 1:
        # Not fatal here so the failure is reported by preflight, against the
        # driver, in the same shape as every other distributed refusal.
        log.error(
            "could not read the packaged dist_harness source; this %d-rank task has no harness to "
            "measure inside and its driver will be refused",
            nproc_per_node,
        )
    reference_note = reference_prefix + _reference_note(ref_dir, agent_workspace)

    def _open_scaffold() -> None:
        """Put the authoring-only reference bundle back for the next attempt."""
        nonlocal reference_note
        if ref_dir is None or ref_dir.is_dir():
            return
        materialized = _materialize_reference(workspace)
        if materialized is None:
            # The note enumerates the contract and the reference drivers by path, in a prompt that also tells the
            # agent not to rely on memory, so keeping it would send this attempt to Read files that are gone.
            log.warning(
                "could not re-materialize the authoring reference bundle at %s; "
                "this attempt's prompt drops its file list",
                ref_dir,
            )
        reference_note = reference_prefix + _reference_note(materialized, agent_workspace)

    def _reset_scaffold() -> None:
        # Put the workspace into the exact state the driver will be judged in — which is also the state the loop's
        # baseline will run it in.
        if external_transaction is not None:
            _restore_kernel_workspace()
            external_transaction.restore_passthroughs()
        else:
            _restore_protected()
        _safe_rmtree(ref_dir)
        if ref_dir is not None and ref_dir.exists():
            # A partial removal leaves both halves of the invariant broken at once and neither is visible later:
            # preflight judges the driver against scaffolding the prep commit then deletes, and `git add -A` carries
            # what survived into the pristine commit.
            raise ScaffoldRetirementError(
                f"could not retire the authoring reference bundle at {ref_dir}; the "
                "driver would be validated against scaffolding that preparation "
                "then deletes, and the surviving files would enter the pristine "
                "commit"
            )
        if provided_harness:
            harness_path.write_text(canonical_harness)
        if canonical_dist_harness is not None:
            dist_harness_path.write_text(canonical_dist_harness, encoding="utf-8")
        if spec_path is not None and canonical_spec:
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(canonical_spec, encoding="utf-8")

    async def _finish_success(
        pf: PreflightResult,
        attempt_count: int,
    ) -> PrepareResult:
        if driver_external:
            # Publish the complete validated driver/helper change set from the isolated staging tree.
            _restore_kernel_workspace()
            assert external_transaction is not None
            external_transaction.restore_passthroughs()
        elif provided_harness:
            # Drop an unused provided harness so it does not become persistent scaffolding when the driver does not
            # import it.
            try:
                uses_harness = "graph_harness" in driver_path.read_text()
            except OSError:
                uses_harness = True
            if not uses_harness:
                _safe_unlink(harness_path)
        if unrestored_sources:
            # Signing off here would carry the agent's edit of the source under optimization into the pristine
            # commit, and every measurement after it would judge a kernel the caller never wrote.
            return PrepareResult(
                ok=False,
                attempts=attempt_count,
                wrote_files=[],
                created_files=[],
                rolled_back=False,
                final_preflight=pf,
                message=_unrestored_message(),
                audit_dir=audit_dir_str,
            )
        if driver_external:
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

        # In-repository task scaffolding must become part of pristine before IterationLoop captures its base SHA.
        stage_code, stage_out = _stage_preparation_changes(workspace, pre_untracked)
        if stage_code != 0:
            _rollback()
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
        # The driver is about to become pristine; anything it reads at runtime has to become pristine with it.
        spec_indexed = None if spec_path is None else _git_indexed(workspace, spec_path)
        if spec_path is not None and spec_indexed is not True:
            _rollback()
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
        _, commit_out = _git(
            workspace,
            "commit",
            "-m",
            "forge prepass: prepared measurement driver",
        )
        if _git_head(workspace) == prep_base_sha:
            _rollback()
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
            wrote_files=_git_changed_since(workspace, prep_base_sha),
            created_files=[],
            rolled_back=False,
            final_preflight=pf,
            message="task prepared",
            audit_dir=audit_dir_str,
        )

    start = time.monotonic()
    # The in-loop preflight must respect the preparation wall (deadline_sec), not only the outer per-kernel
    # deadline_unix.
    preflight_deadline_unix = time.time() + deadline_sec
    if deadline_unix > 0:
        preflight_deadline_unix = min(preflight_deadline_unix, deadline_unix)
    attempts = 0
    prior_failure = ""
    prior_failure_heading = RETRY_HEADING_DEFAULT
    last_pf = preflight
    external_rollback_error = ""
    # An attempt that leaves the driver byte-identical is a distinct failure from one that edited it badly, and the
    # two need different guidance.
    edited_any_attempt = False
    starved_retry_sec = 0.0
    scaffold_error = ""
    try:
        while attempts < PREPARE_MAX_ATTEMPTS:
            remaining = deadline_sec - (time.monotonic() - start)
            if remaining <= 10:
                break
            if attempts and remaining < PREPARE_MIN_RETRY_SEC:
                starved_retry_sec = remaining
                break
            attempts += 1
            is_last_attempt = attempts >= PREPARE_MAX_ATTEMPTS
            spendable = max(0.0, remaining - _SALVAGE_RESERVE_SEC)
            attempt_timeout = spendable if is_last_attempt else min(spendable, float(PER_ATTEMPT_CAP_SEC))
            # Each attempt authors against the reference bundle; the preceding attempt's verdict retired it (see
            # _reset_scaffold).
            _open_scaffold()
            prompt = _build_prompt(
                evidence,
                driver_rel,
                reference_note,
                prior_failure,
                invocation_note,
                prior_failure_heading,
                _distributed_contract_note(nproc_per_node),
            )
            attempt_dir = f"attempt_{attempts:02d}"
            _audit_text(f"{attempt_dir}/prompt.md", prompt)
            _audit_text(f"{attempt_dir}/system_prompt.md", _SYSTEM_PROMPT)
            _audit_driver(f"{attempt_dir}/driver_before.py")
            digest_before = _driver_digest()
            progress_log: list[str] = []
            agent_started = time.monotonic()

            try:
                agent_output = await _run_prepare_agent(
                    config=config,
                    workspace=agent_workspace,
                    system_prompt=_SYSTEM_PROMPT,
                    prompt=prompt,
                    timeout_sec=attempt_timeout,
                    additional_dirs=([str(workspace)] if driver_external else None),
                    allow_shell=not driver_external,
                    target_files=[
                        driver_path.as_posix(),
                        harness_path.as_posix(),
                    ],
                    protected_files=backend_protected_files,
                    usage=usage,
                    progress_log=progress_log,
                )
            except asyncio.TimeoutError:
                _audit_driver(f"{attempt_dir}/driver_at_timeout.py")
                elapsed_s = round(time.monotonic() - agent_started, 3)
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "timeout",
                        "elapsed_s": elapsed_s,
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                _reset_scaffold()
                # The Agent may have completed a valid driver before getting stuck on self-verification.
                last_pf = await _preflight_async(
                    driver_path.as_posix(),
                    snr_threshold,
                    PREFLIGHT_WARMUP,
                    PREFLIGHT_ITERS,
                    require_graph=True,
                    require_profile=True,
                    require_ranks=nproc_per_node,
                    deadline_unix=preflight_deadline_unix,
                    expected_case_ids=expected_case_ids,
                )
                _audit_driver(f"{attempt_dir}/driver_after_timeout_preflight.py")
                _audit_json(f"{attempt_dir}/preflight.json", asdict(last_pf))
                if last_pf.ok:
                    return await _finish_success(last_pf, attempts)
                prior_failure_heading = RETRY_HEADING_DEFAULT if driver_edited else RETRY_HEADING_NO_EDIT
                if driver_edited:
                    jit_hint = ""
                    if last_pf.all_failures_are_timeouts:
                        jit_hint = (
                            "\nNOTE: Every failure above is a TIMEOUT, not a crash. "
                            "This usually means the driver is structurally correct "
                            "but the first execution triggered slow JIT compilation "
                            "(CK/aiter kernels can take 44s+ per module). Do NOT "
                            "rewrite the driver from scratch — verify it is "
                            "structurally correct and resubmit; the next preflight "
                            "run benefits from a warm JIT cache.\n"
                        )
                    prior_failure = (
                        "Agent timed out, then deterministic preflight failed:\n" + last_pf.detail_report() + jit_hint
                    )
                else:
                    prior_failure = (
                        f"Your previous attempt ran for {elapsed_s:.0f}s and timed out "
                        f"having made NO edit at all to `{driver_rel}` — the file is "
                        "byte-identical to before. Reading and planning is not "
                        "progress here. Open the driver and WRITE the fix as your "
                        "first substantive action, then verify; if you cannot finish "
                        "the whole contract in the time you have, still leave the "
                        "best driver you can on disk rather than nothing.\n"
                        f"What you spent that time on — {summarize_agent_progress(progress_log)}\n"
                        "The deterministic check on that unchanged driver reported:\n" + last_pf.detail_report()
                    )
                remaining_after_timeout = deadline_sec - (time.monotonic() - start)
                if attempts < PREPARE_MAX_ATTEMPTS:
                    if remaining_after_timeout >= PREPARE_MIN_RETRY_SEC:
                        continue
                    if remaining_after_timeout > 10:
                        starved_retry_sec = remaining_after_timeout
                break
            except Exception as exc:  # noqa: BLE001
                _audit_driver(f"{attempt_dir}/driver_at_exception.py")
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "error",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "elapsed_s": round(time.monotonic() - agent_started, 3),
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                prior_failure = f"Agent invocation error: {type(exc).__name__}: {exc}"
                prior_failure_heading = RETRY_HEADING_DEFAULT
                _reset_scaffold()
                continue
            else:
                _audit_text(f"{attempt_dir}/agent_output.txt", str(agent_output or ""))
                _audit_text(
                    f"{attempt_dir}/agent_progress.txt",
                    "\n".join(progress_log),
                )
                _audit_driver(f"{attempt_dir}/driver_after.py")
                driver_edited = _detect_driver_edited(digest_before)
                edited_any_attempt = edited_any_attempt or driver_edited
                _audit_json(
                    f"{attempt_dir}/agent_event.json",
                    {
                        "status": "completed",
                        "elapsed_s": round(time.monotonic() - agent_started, 3),
                        "budget_s": round(attempt_timeout, 3),
                        "driver_edited": driver_edited,
                    },
                )

            # (1) Source protection: whatever the agent did, restore source (and the provided harness) to pristine
            # before we judge the driver.
            _reset_scaffold()

            # require_graph=True: a produced driver must actually time under a CUDA/HIP graph, not eagerly (nor
            # silently fall back to eager).
            last_pf = await _preflight_async(
                driver_path.as_posix(),
                snr_threshold,
                PREFLIGHT_WARMUP,
                PREFLIGHT_ITERS,
                require_graph=True,
                require_profile=True,
                require_ranks=nproc_per_node,
                deadline_unix=preflight_deadline_unix,
                expected_case_ids=expected_case_ids,
            )
            _audit_driver(f"{attempt_dir}/driver_at_preflight.py")
            _audit_json(f"{attempt_dir}/preflight.json", asdict(last_pf))
            if last_pf.ok:
                return await _finish_success(last_pf, attempts)
            prior_failure_heading = RETRY_HEADING_DEFAULT if driver_edited else RETRY_HEADING_NO_EDIT
            jit_hint = ""
            if driver_edited and last_pf.all_failures_are_timeouts:
                jit_hint = (
                    "\nNOTE: Every failure above is a TIMEOUT, not a crash. "
                    "This usually means the driver is structurally correct "
                    "but the first execution triggered slow JIT compilation "
                    "(CK/aiter kernels can take 44s+ per module). Do NOT "
                    "rewrite the driver from scratch — verify it is "
                    "structurally correct and resubmit; the next preflight "
                    "run benefits from a warm JIT cache.\n"
                )
            prior_failure = (
                (
                    "Deterministic preflight after your edit:\n"
                    if driver_edited
                    else (
                        f"You finished without editing `{driver_rel}` at all. The "
                        "deterministic check therefore ran the SAME driver again:\n"
                    )
                )
                + last_pf.detail_report()
                + jit_hint
            )
    except ScaffoldRetirementError as exc:
        scaffold_error = str(exc)
    finally:
        # Never leave reference or external staging bundles behind.
        _safe_rmtree(ref_dir)
        if external_transaction is not None:
            if not external_transaction.published:
                try:
                    external_transaction.rollback()
                except ExternalArtifactError as exc:
                    external_rollback_error = str(exc)
            if not external_transaction.published:
                _rollback()
            try:
                external_transaction.close()
            except OSError as exc:
                if not external_rollback_error:
                    external_rollback_error = f"staging cleanup failed: {exc}"

    # (3) Failure: roll the workspace back to its exact pre-prep state (tracked files reset to HEAD, caller's
    # uncommitted mods re-applied, prep-created untracked removed, protected source restored).
    _rollback()
    rolled_back = not external_rollback_error and not unrestored_sources
    if unrestored_sources:
        # Lead with it: the operator has a file on disk that is not theirs any more, which they have to deal
        # with before any reason the driver was refused.
        log.error("%s", _unrestored_message())
        return PrepareResult(
            ok=False,
            attempts=attempts,
            wrote_files=[],
            created_files=[],
            rolled_back=rolled_back,
            final_preflight=last_pf,
            message=_unrestored_message(),
            audit_dir=audit_dir_str,
        )
    if scaffold_error:
        # Lead with it: the preflight reasons describe a driver judged in a state that was never valid, so quoting
        # them first would send the operator after the driver.
        log.error("%s", scaffold_error)
        return PrepareResult(
            ok=False,
            attempts=attempts,
            wrote_files=[],
            created_files=[],
            rolled_back=rolled_back,
            final_preflight=last_pf,
            message=scaffold_error,
            audit_dir=audit_dir_str,
        )
    return PrepareResult(
        ok=False,
        attempts=attempts,
        wrote_files=[],
        created_files=[],
        rolled_back=rolled_back,
        final_preflight=last_pf,
        message=(
            (
                f"prep wall exhausted after {attempts} attempt(s); the remaining "
                f"{starved_retry_sec:.0f}s is below the {PREPARE_MIN_RETRY_SEC}s "
                "minimum retry budget, so no further attempt was started — raise "
                "the per-kernel deadline to give preparation more room"
            )
            if starved_retry_sec
            else "could not produce a conforming driver within the budget"
            if edited_any_attempt or not attempts
            else (
                f"prep agent never edited the driver in {attempts} attempt(s); "
                "the driver is unchanged, so the preflight reasons below describe "
                "the ORIGINAL driver, not a failed repair"
            )
        )
        + (f"; external artifact rollback failed: {external_rollback_error}" if external_rollback_error else ""),
        audit_dir=audit_dir_str,
    )


def prepare_task_sync(**kwargs) -> PrepareResult:
    """Synchronous wrapper for CLI code."""

    return asyncio.run(prepare_task(**kwargs))
