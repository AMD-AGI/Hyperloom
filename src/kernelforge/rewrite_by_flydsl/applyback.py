# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a framework-level patch from the best verified FlyDSL rewrite."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kernelforge.tools.git import git, git_bytes
from kernelforge.agent_backends import (
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.config import Config
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.rewrite_by_flydsl import protocol
from kernelforge.rewrite_by_flydsl.budget import DEFAULT_REWRITE_BUDGET
from kernelforge.rewrite_by_flydsl.measurement import FlydslMeasurement
from kernelforge.rewrite_by_flydsl.protocol import validate_applyback_manifest
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec
from kernelforge.tools.durable_io import atomic_write_json, atomic_write_text
from kernelforge.tracker import UsageAccumulator

log = logging.getLogger(__name__)

# Framework apply-back artifacts live beside, never inside, the artifact paths the nested standalone FlyDSL forge-loop
# owns (``forge_experiments/best*``).
APPLYBACK_NAMESPACE = "rewrite_applyback"
_IMPORT_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_NO_TIME_FOR_APPLYBACK = "insufficient time remaining for the apply-back agent and host validation"
_UNRETRYABLE_FAILURE_MARKERS = ("baseline apply-back import validation", "insufficient time remaining")


@dataclass(frozen=True)
class ApplybackRequest:
    """One apply-back: the pristine base, the best and what it measured, diagnostics, deadline, imports, attempts."""

    base_commit: str
    experiments_dir: str
    best_commit: str = ""
    measurement: FlydslMeasurement = FlydslMeasurement(None, None, None)
    deadline_unix: float | None = None
    import_modules: list[str] | tuple[str, ...] = ()
    max_attempts: int = 2


@dataclass(frozen=True)
class CollectedPatch:
    """The integration commit, the patch from the pristine base to it, the files it changes and the ref keeping it."""

    base_commit: str
    applyback_commit: str
    commit_ref: str
    patch: str
    changed_files: list[str]


@dataclass(frozen=True)
class _IntegrationBrief:
    """What every attempt integrates: the rewrite, its framework, the source path and the read-only reference."""

    spec: RewriteSpec
    framework: str
    source_relative: str
    reference_path: Path


@dataclass(frozen=True)
class _ApplybackTurn:
    """One clean-room session: its worktree, budget, progress log and the failure the previous attempt left."""

    worktree: Path
    timeout_sec: int
    progress_log: list[str]
    prior_failure: str = ""


@dataclass
class ApplybackResult:
    ok: bool
    patch_path: str = ""
    manifest_path: str = ""
    changed_files: list[str] = field(default_factory=list)
    error: str = ""
    agent_backend: str = ""
    agent_model: str = ""
    base_commit: str = ""
    best_commit: str = ""
    commit_ref: str = ""
    diagnostic_path: str = ""
    canonical_patch_path: str = ""
    canonical_files_root: str = ""
    canonical_result_path: str = ""
    forge_workspace: str = ""
    artifacts: list[str] = field(default_factory=list)
    import_validation_modules: list[str] = field(default_factory=list)
    attempts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _git(workspace: str | Path, *args: str) -> subprocess.CompletedProcess:
    return git("-C", str(workspace), *args, check=False)


def _infer_framework(spec: RewriteSpec) -> str:
    if spec.framework.strip():
        return spec.framework.strip().lower()
    parts = {part.lower() for part in Path(spec.source_kernel).parts}
    for candidate in protocol.SUPPORTED_FRAMEWORKS:
        if candidate in parts:
            return candidate
    return "unknown"


@dataclass(frozen=True)
class ImportValidationPlan:
    """Import targets and worktree roots used before and after apply-back."""

    modules: tuple[str, ...]
    python_roots: tuple[str, ...]


def _infer_source_module(worktree: Path, source_relative: str) -> str:
    source = worktree / source_relative
    if source.suffix != ".py":
        raise RuntimeError(f"apply-back import validation requires a Python source module: {source_relative}")
    parts = [] if source.name == "__init__.py" else [source.stem]
    package = source.parent
    while (package / "__init__.py").is_file():
        parts.insert(0, package.name)
        package = package.parent
    if not parts:
        raise RuntimeError(f"could not infer an import module from package initializer: {source_relative}")
    return ".".join(parts)


def _build_import_validation_plan(
    *,
    worktree: Path,
    source_relative: str,
    import_modules: list[str] | tuple[str, ...],
) -> ImportValidationPlan:
    """Resolve explicit targets or infer the source module without framework rules."""

    requested = [str(module).strip() for module in import_modules if str(module).strip()]
    modules = requested or [_infer_source_module(worktree, source_relative)]
    invalid = [module for module in modules if not _IMPORT_MODULE_RE.fullmatch(module)]
    if invalid:
        raise RuntimeError("invalid apply-back import module name: " + ", ".join(invalid))

    source_parent = (worktree / source_relative).resolve().parent
    roots: list[str] = []
    current = source_parent
    while current == worktree or worktree in current.parents:
        text = str(current)
        if text not in roots:
            roots.append(text)
        if current == worktree:
            break
        current = current.parent
    return ImportValidationPlan(
        modules=tuple(dict.fromkeys(modules)),
        python_roots=tuple(roots),
    )


def _validate_imports(
    *,
    worktree: Path,
    plan: ImportValidationPlan,
    timeout_sec: int,
    stage: str,
) -> None:
    """Import every target in a fresh isolated interpreter."""

    started = time.monotonic()
    for module in plan.modules:
        remaining = timeout_sec - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError(f"{stage} apply-back import validation timed out")
        script = (
            "import importlib, json, sys;"
            "sys.dont_write_bytecode = True;"
            f"sys.path[:0] = json.loads({json.dumps(json.dumps(plan.python_roots))});"
            f"importlib.import_module({module!r});"
            f"print('import_ok: {module}')"
        )
        try:
            checked = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=max(1, int(remaining)),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{stage} apply-back import validation timed out for {module}") from error
        if checked.returncode != 0:
            output = (checked.stderr or checked.stdout or "").strip()[-2000:]
            raise RuntimeError(f"{stage} apply-back import validation failed for {module}: {output}")


def _build_prompt(brief: _IntegrationBrief, *, time_budget_seconds: int, prior_failure: str = "") -> str:
    spec = brief.spec
    targets = ", ".join(spec.target_functions) or "(not specified)"
    retry_context = (
        "\n## Previous clean-room attempt\n"
        "The previous attempt was discarded after host validation failed:\n"
        f"```\n{prior_failure[-3000:]}\n```\n"
        "Start again from this pristine base and correct that failure.\n"
        if prior_failure
        else ""
    )
    return f"""\
Integrate a verified FlyDSL rewrite into the original {brief.framework} repository.

Repository source operation:
- Source file: `{brief.source_relative}`
- Source symbols: `{targets}`
- Logical operator: `{spec.op_name}`
- Required FlyDSL factory: `{spec.builder_symbol}`

The latest correctness-verified and performance-selected standalone FlyDSL
implementation is available read-only at:
`{brief.reference_path}`

Inspect the repository and implement the production integration now. Use the
reference implementation as the computational source of truth, but place code
in the framework's normal source tree and update every required dispatch,
registration, binding, build, packaging, and import surface so the framework's
existing public operator path can select and execute the FlyDSL implementation.
{retry_context}

## Convergence contract
- Your total session budget is {time_budget_seconds} seconds.
- Spend the first part inspecting and editing. Reserve the final
  {DEFAULT_REWRITE_BUDGET.applyback_agent_finalization_reserve_sec} seconds to
  review `git diff`, run one focused import/targeted test if needed, and finish.
- The standalone kernel has already passed correctness and performance gates.
  Do NOT benchmark, profile, tune, or test unrelated shapes.
- Do NOT run broad test directories or full suites. At most run one focused
  existing operator test and one import smoke check. The caller performs fixed
  syntax, pre-commit, patch, and `git apply --check` validation after you return.
- Once the integration and focused check are complete, stop immediately with a
  concise summary. Do not continue exploring optional cleanup or documentation.

Rules:
- Preserve the framework's public API and fallback behavior.
- Follow existing repository conventions; do not introduce a one-off loader.
- Prefer a local lazy dispatch seam. Do not add a framework-wide backend enum or
  a new environment variable unless the existing framework contract requires it.
- Do not edit tests, benchmarks, measurement drivers, generated artifacts, or
  the read-only reference file.
- Do not commit or change branches.
- Make only changes required for this integration.
- The caller will export your working-tree changes as a git-apply patch and run
  framework integration validation later. Finish with a concise summary.
"""


def _make_applyback_hooks(*, deadline_monotonic: float) -> AgentHooks:
    """Bound shell work and stop all tools during finalization reserve."""

    async def _bound_bash(input_data: dict, tool_use_id, context) -> dict:
        remaining = deadline_monotonic - time.monotonic()
        finalization_reserve = DEFAULT_REWRITE_BUDGET.applyback_agent_finalization_reserve_sec
        if remaining <= finalization_reserve:
            reason = (
                "Apply-back finalization reserve has started. Stop running "
                "tools and return your concise integration summary now."
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        if input_data.get("tool_name") != "Bash":
            return {}
        tool_input = input_data.get("tool_input") or {}
        command = str(tool_input.get("command") or "")
        lowered = command.lower()
        if re.search(r"\b(benchmark|bench|rocprof|nsys|profile)\b", lowered):
            reason = (
                "Apply-back receives an already benchmarked FlyDSL kernel. "
                "Do not benchmark or profile; finish the framework integration."
            )
        else:
            allowed_sec = max(
                0,
                min(
                    DEFAULT_REWRITE_BUDGET.applyback_shell_command_max_sec,
                    int(remaining - finalization_reserve),
                ),
            )
            requested_ms = tool_input.get("timeout")
            shell_timeout = re.search(
                r"(?:^|[;&|]\s*)timeout\s+(\d+)([smh]?)\b",
                command,
            )
            requested_sec = 0
            if isinstance(requested_ms, (int, float)):
                requested_sec = int(float(requested_ms) / 1000.0)
            if shell_timeout:
                value = int(shell_timeout.group(1))
                unit = shell_timeout.group(2)
                requested_sec = max(
                    requested_sec,
                    value * {"": 1, "s": 1, "m": 60, "h": 3600}[unit],
                )
            if requested_sec > allowed_sec:
                reason = (
                    f"This command requests up to {requested_sec}s, but only "
                    f"{allowed_sec}s of tool budget remains. Use a narrower check "
                    "within that limit, or finish now."
                )
            else:
                return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return AgentHooks(
        pre_tool_use=[
            AgentHook(
                matcher="",
                callback=_bound_bash,
                timeout_sec=5,
            )
        ]
    )


async def _run_agent(
    *,
    config: Config,
    brief: _IntegrationBrief,
    turn: _ApplybackTurn,
    usage: UsageAccumulator | None = None,
) -> tuple[str, str]:
    worktree = turn.worktree
    timeout_sec = turn.timeout_sec
    runtime = agent_runtime(config)
    backend = create_registered_backend(
        runtime,
        preflight=False,
        probe_cwd=str(worktree),
    )
    prompt = _build_prompt(brief, time_budget_seconds=timeout_sec, prior_failure=turn.prior_failure)
    deadline_monotonic = time.monotonic() + timeout_sec
    run_spec = AgentRunSpec(
        system_prompt=(
            "You are a senior GPU framework integration engineer. Produce a "
            "maintainable repository-level integration from a verified FlyDSL "
            "reference implementation. Work directly in the supplied git worktree."
        ),
        user_prompt=prompt,
        role="flydsl applyback",
        cwd=str(worktree),
        writable=True,
        timeout_sec=timeout_sec,
        additional_directories=[str(brief.reference_path.parent)],
        allow_untracked=True,
        protected_path_rules=FORGE_PROTECTED_PATHS,
        hooks=_make_applyback_hooks(deadline_monotonic=deadline_monotonic),
        progress_log=turn.progress_log,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=min(config.max_turns, 40),
            bare=True,
        ),
    )
    result = await asyncio.wait_for(
        backend.run(run_spec, usage=usage),
        timeout=watchdog_timeout_sec(timeout_sec),
    )
    # A turn cap or SDK error leaves a half-rewired integration that passes host validation and every gate after it,
    # so it must be raised rather than published.
    if result.end_reason != "agent_stopped":
        raise RuntimeError(f"apply-back agent did not finish normally: {result.end_reason or 'unknown'}")
    return backend.name, backend.runtime.model


def _publishable_staged_paths(worktree: Path, empty_error: str) -> list[str]:
    """The staged change set, refused when it is empty or carries what a framework patch must never carry."""
    names_raw = _git(worktree, "diff", "--cached", "--name-only", "-z", "--", ".").stdout
    changed_files = [path for path in names_raw.split("\0") if path]
    if not changed_files:
        raise RuntimeError(empty_error)
    forbidden = [path for path in changed_files if path.startswith(("test/", "tests/", "benchmark/", "benchmarks/"))]
    if forbidden:
        raise RuntimeError("apply-back agent modified protected validation files: " + ", ".join(forbidden))
    producer_owned = [path for path in changed_files if protocol.is_producer_owned_path(path)]
    if producer_owned:
        raise RuntimeError(
            "apply-back agent would publish producer-owned forge state as a "
            "framework change: " + ", ".join(producer_owned)
        )
    return changed_files


def _validate_worktree_changes(
    *,
    worktree: Path,
    timeout_sec: int,
    import_plan: ImportValidationPlan | None = None,
) -> list[str]:
    """Run fixed host-owned checks before exporting an apply-back patch."""
    validation_started = time.monotonic()
    staged = _git(worktree, "add", "-A", "--", ".")
    if staged.returncode != 0:
        raise RuntimeError(f"could not stage integration changes: {staged.stderr.strip()}")
    changed_files = _publishable_staged_paths(worktree, "apply-back agent produced no repository changes")
    checked = _git(worktree, "diff", "--cached", "--check")
    if checked.returncode != 0:
        raise RuntimeError(f"apply-back diff check failed: {checked.stdout.strip()}")

    python_files = [worktree / path for path in changed_files if path.endswith(".py") and (worktree / path).is_file()]
    for python_file in python_files:
        try:
            compile(python_file.read_text(encoding="utf-8"), str(python_file), "exec", dont_inherit=True)
        except (OSError, SyntaxError) as error:
            raise RuntimeError(f"apply-back Python syntax validation failed for {python_file}: {error}") from error

    precommit_config = worktree / ".pre-commit-config.yaml"
    precommit = shutil.which("pre-commit")
    if precommit and precommit_config.is_file():
        for _ in range(2):
            remaining = timeout_sec - (time.monotonic() - validation_started)
            if remaining <= 0:
                raise RuntimeError("apply-back pre-commit validation timed out")
            try:
                checked = subprocess.run(
                    [precommit, "run", "--files", *changed_files],
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                    timeout=max(1, min(remaining, 300)),
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("apply-back pre-commit validation timed out") from error
            # Formatting hooks conventionally return 1 after fixing files.
            staged = _git(worktree, "add", "-A", "--", ".")
            if staged.returncode != 0:
                raise RuntimeError(f"could not restage pre-commit changes: {staged.stderr.strip()}")
            if checked.returncode == 0:
                break
        else:
            raise RuntimeError(f"apply-back pre-commit validation failed: {(checked.stdout or checked.stderr)[-2000:]}")
        checked = _git(worktree, "diff", "--cached", "--check")
        if checked.returncode != 0:
            raise RuntimeError(f"post-format apply-back diff check failed: {checked.stdout.strip()}")
        # Hooks can add files of their own or normalize an edit back to its committed state, so the final staged set
        # is what the patch will carry.
        changed_files = _publishable_staged_paths(
            worktree, "apply-back pre-commit hooks reverted every repository change"
        )
    if import_plan is not None:
        remaining = timeout_sec - (time.monotonic() - validation_started)
        if remaining <= 0:
            raise RuntimeError("patched apply-back import validation timed out")
        _validate_imports(worktree=worktree, plan=import_plan, timeout_sec=max(1, int(remaining)), stage="patched")
    return changed_files


def _snapshot_failure(
    *,
    worktree: Path,
    experiments_dir: str,
    error: str,
    progress_log: list[str],
    attempt: int = 1,
) -> str:
    """Preserve an explicitly non-publishable diagnostic patch on failure."""
    root = (
        Path(experiments_dir).resolve() / APPLYBACK_NAMESPACE / "failed" / f"attempt_{attempt:02d}_{int(time.time())}"
    )
    root.mkdir(parents=True, exist_ok=True)
    _git(worktree, "add", "-A", "--", ".")
    partial = _git(worktree, "diff", "--cached", "--binary", "--", ".")
    status = _git(worktree, "status", "--short")
    atomic_write_text(root / "partial.patch", partial.stdout or "")
    atomic_write_text(root / "status.txt", status.stdout or "")
    atomic_write_text(root / "error.txt", error + "\n")
    atomic_write_json(root / "progress.json", {"events": progress_log})
    return str(root)


def _collect_patch(spec: RewriteSpec, worktree: Path, patch_path: Path, base_commit: str) -> CollectedPatch:
    staged = _git(worktree, "add", "-A", "--", ".")
    if staged.returncode != 0:
        raise RuntimeError(f"could not stage integration changes: {staged.stderr.strip()}")
    committed = _git(
        worktree,
        "-c",
        "user.name=forge-rewrite",
        "-c",
        "user.email=forge-rewrite@local",
        "commit",
        "-m",
        f"forge-rewrite: integrate {spec.op_name} flydsl apply-back",
    )
    if committed.returncode != 0:
        raise RuntimeError(
            "could not commit integration patch in the temporary worktree: "
            f"{(committed.stderr or committed.stdout).strip()}"
        )
    applyback_commit = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    diff = _git(
        worktree,
        "diff",
        base_commit,
        applyback_commit,
        "--binary",
        "--no-ext-diff",
        "--",
        ".",
    )
    if diff.returncode != 0:
        raise RuntimeError(f"could not export integration patch: {diff.stderr.strip()}")
    patch = diff.stdout
    if not patch.strip():
        raise RuntimeError("apply-back agent produced no repository changes")

    names_raw = _git(
        worktree,
        "diff",
        "--name-only",
        "-z",
        base_commit,
        applyback_commit,
        "--",
        ".",
    ).stdout
    changed_files = [path for path in names_raw.split("\0") if path]
    atomic_write_text(patch_path, patch)

    # Verify against the pristine base, not against the agent's already-modified worktree.
    reset = _git(worktree, "reset", "--hard", base_commit)
    if reset.returncode != 0:
        raise RuntimeError(f"could not reset patch verification worktree: {reset.stderr.strip()}")
    clean = _git(worktree, "clean", "-fd")
    if clean.returncode != 0:
        raise RuntimeError(f"could not clean patch verification worktree: {clean.stderr.strip()}")
    checked = _git(worktree, "apply", "--check", str(patch_path))
    if checked.returncode != 0:
        raise RuntimeError(f"generated framework patch does not apply to the pristine base: {checked.stderr.strip()}")
    commit_ref = f"refs/forge-rewrite/applyback/{spec.operator_slug}-{applyback_commit[:12]}"
    published_ref = _git(Path(spec.workspace).resolve(), "update-ref", commit_ref, applyback_commit)
    if published_ref.returncode != 0:
        raise RuntimeError(f"could not preserve apply-back commit: {published_ref.stderr.strip()}")
    return CollectedPatch(base_commit, applyback_commit, commit_ref, patch, changed_files)


def _publish_patch(
    spec: RewriteSpec,
    framework: str,
    flydsl_best_commit: str,
    measurement: FlydslMeasurement,
    collected: CollectedPatch,
) -> tuple[str, str, str, str]:
    """Publish the framework patch into the apply-back-owned artifact namespace."""
    applyback_commit = collected.applyback_commit
    changed_files = collected.changed_files
    workspace = Path(spec.workspace).resolve()
    root = workspace / "forge_experiments"
    namespace_root = root / APPLYBACK_NAMESPACE
    best_root = namespace_root / "best"
    manifest_path = best_root / "manifest.json"
    result_path = namespace_root / "result.json"
    previous_iteration = -1
    for previous_path in (manifest_path, result_path):
        try:
            previous = json.loads(previous_path.read_text())
            previous_iteration = max(
                previous_iteration,
                int(previous.get("iteration", -1)),
            )
        except (OSError, ValueError, TypeError):
            # A missing or corrupt prior manifest simply starts at iteration 0.
            continue
    iteration = previous_iteration + 1
    while (best_root / f"iter_{iteration:03d}").exists():
        iteration += 1
    version_name = f"iter_{iteration:03d}"
    version = best_root / version_name
    relative_dir = version.relative_to(root)
    manifest = validate_applyback_manifest(
        {
            "schema_version": protocol.ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": protocol.ARTIFACT_KIND_FRAMEWORK_APPLYBACK,
            "validation_scope": protocol.VALIDATION_SCOPE_REFERENCE,
            "iteration": iteration,
            "logical_op_name": spec.op_name,
            "operator_slug": spec.operator_slug,
            "builder_symbol": spec.builder_symbol,
            "source_entry": spec.source_entry,
            "commit_hash": applyback_commit,
            "base_commit": collected.base_commit,
            "commit_ref": collected.commit_ref,
            "flydsl_best_commit": flydsl_best_commit,
            "baseline_wall_ms": measurement.source_ms,
            "best_wall_ms": measurement.flydsl_ms,
            "speedup": measurement.speedup,
            "reference_correctness_passed": True,
            "reference_snr_db": measurement.snr_db,
            "integration_validation_required": True,
            "integration_validation_status": protocol.INTEGRATION_VALIDATION_PENDING,
            "target_language": "flydsl",
            "framework": framework,
            "changed_files": changed_files,
            "artifact_dir": relative_dir.as_posix(),
            "patch_path": (relative_dir / "forge.patch").as_posix(),
            "validation_path": (relative_dir / "validation.txt").as_posix(),
            "benchmark_path": (relative_dir / "benchmark.json").as_posix(),
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    best_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=str(best_root),
            prefix=f".{version_name}.",
        )
    )
    try:
        atomic_write_text(temporary / "forge.patch", collected.patch)
        atomic_write_text(
            temporary / "validation.txt",
            "Standalone FlyDSL reference passed the rewrite correctness gate.\n"
            "Framework integration validation is intentionally pending in Hyperloom.\n",
        )
        atomic_write_json(
            temporary / "benchmark.json",
            {
                "source_ms": measurement.source_ms,
                "flydsl_best_ms": measurement.flydsl_ms,
            },
        )
        atomic_write_json(temporary / "publication.json", manifest)
        files_root = temporary / "files"
        files_root.mkdir(parents=True, exist_ok=True)
        for relative in changed_files:
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise RuntimeError(f"apply-back changed file escapes repository: {relative}")
            content = git_bytes("-C", str(workspace), "show", f"{applyback_commit}:{target.as_posix()}", check=False)
            # Deleted files belong in the patch but have no final snapshot.
            if content.returncode != 0:
                continue
            destination = files_root / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content.stdout)
        os.replace(temporary, version)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    # The bundle is complete on disk before either pointer becomes readable, so a hard kill can only leave the
    # previous publication or nothing at all.
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(result_path, manifest)
    return (
        str(version / "forge.patch"),
        str(manifest_path),
        str(version / "files"),
        str(result_path),
    )


@dataclass(frozen=True)
class _CleanRoomAttempts:
    """The bounded clean-room attempts of one apply-back: its inputs and the brief every attempt integrates."""

    config: Config
    request: ApplybackRequest
    brief: _IntegrationBrief
    usage: UsageAccumulator | None

    def run(self) -> ApplybackResult:
        """Attempt until one publishes, one fails in a way a fresh attempt cannot fix, or time runs out."""
        request, budget = self.request, DEFAULT_REWRITE_BUDGET
        workspace = Path(self.brief.spec.workspace).resolve()
        source_relative = self.brief.source_relative
        prior_failure = ""
        last_result = ApplybackResult(ok=False, error=_NO_TIME_FOR_APPLYBACK)
        for attempt in range(1, request.max_attempts + 1):
            if not budget.can_start_applyback(request.deadline_unix):
                break
            worktree = Path(tempfile.mkdtemp(prefix="forge_rewrite_applyback_worktree_"))
            worktree_added = False
            progress_log: list[str] = []
            timeout_sec = 0
            diagnostic_path = ""
            error_text = ""
            try:
                added = _git(workspace, "worktree", "add", "--detach", str(worktree), request.base_commit)
                if added.returncode != 0:
                    return ApplybackResult(
                        ok=False,
                        error=f"could not create pristine apply-back worktree: {added.stderr.strip()}",
                        attempts=attempt,
                    )
                worktree_added = True
                if not (worktree / source_relative).is_file():
                    return ApplybackResult(
                        ok=False,
                        error=f"source kernel is not tracked by the pristine framework commit: {source_relative}",
                        attempts=attempt,
                    )
                import_plan = _build_import_validation_plan(
                    worktree=worktree, source_relative=source_relative, import_modules=request.import_modules
                )
                remaining = budget.remaining_seconds(request.deadline_unix)
                baseline_timeout = int(min(budget.import_validation_timeout_sec, max(1, remaining)))
                _validate_imports(worktree=worktree, plan=import_plan, timeout_sec=baseline_timeout, stage="baseline")
                if not budget.can_start_applyback(request.deadline_unix):
                    raise RuntimeError(
                        "insufficient time remaining after baseline import "
                        "validation for the apply-back agent and host validation"
                    )
                timeout_sec = budget.agent_timeout_sec(
                    deadline_unix=request.deadline_unix,
                    configured_timeout_sec=self.config.agent_timeout_sec,
                    attempts_left=request.max_attempts - attempt + 1,
                )
                turn = _ApplybackTurn(worktree, timeout_sec, progress_log, prior_failure)
                return self._integrate(turn, import_plan, attempt)
            except (asyncio.TimeoutError, TimeoutError):
                error_text = f"apply-back agent timed out after {timeout_sec}s"
            except Exception as error:  # isolation boundary: one apply-back attempt
                log.warning("apply-back attempt %d failed: %s", attempt, error, exc_info=True)
                error_text = f"{type(error).__name__}: {error}"
            finally:
                if error_text and worktree_added:
                    diagnostic_path = _snapshot_failure(
                        worktree=worktree,
                        experiments_dir=request.experiments_dir,
                        error=error_text,
                        progress_log=progress_log,
                        attempt=attempt,
                    )
                if worktree_added:
                    _git(workspace, "worktree", "remove", "--force", str(worktree))
                shutil.rmtree(worktree, ignore_errors=True)

            last_result = ApplybackResult(
                ok=False,
                error=error_text,
                base_commit=request.base_commit,
                diagnostic_path=diagnostic_path,
                attempts=attempt,
            )
            if any(marker in error_text for marker in _UNRETRYABLE_FAILURE_MARKERS):
                break
            prior_failure = error_text
        return last_result

    def _integrate(self, turn: _ApplybackTurn, import_plan: ImportValidationPlan, attempt: int) -> ApplybackResult:
        """Let the agent integrate the reference in the turn's worktree, then validate and publish what it changed."""
        spec, request, budget = self.brief.spec, self.request, DEFAULT_REWRITE_BUDGET
        backend_name, backend_model = asyncio.run(
            _run_agent(config=self.config, brief=self.brief, turn=turn, usage=self.usage)
        )
        if budget.remaining_seconds(request.deadline_unix) <= budget.applyback_post_agent_reserve_sec:
            raise RuntimeError("apply-back agent returned without enough time for host validation")
        host_timeout_sec = budget.host_validation_timeout_sec(request.deadline_unix)
        _validate_worktree_changes(worktree=turn.worktree, timeout_sec=host_timeout_sec, import_plan=import_plan)
        temporary_patch = self.brief.reference_path.parent / f"forge_attempt_{attempt:02d}.patch"
        collected = _collect_patch(spec, turn.worktree, temporary_patch, request.base_commit)
        patch_path, manifest_path, files_root, result_path = _publish_patch(
            spec, self.brief.framework, request.best_commit, request.measurement, collected
        )
        return ApplybackResult(
            ok=True,
            patch_path=patch_path,
            manifest_path=manifest_path,
            changed_files=collected.changed_files,
            agent_backend=backend_name,
            agent_model=backend_model,
            base_commit=request.base_commit,
            best_commit=collected.applyback_commit,
            commit_ref=collected.commit_ref,
            canonical_patch_path=patch_path,
            canonical_files_root=files_root,
            canonical_result_path=result_path,
            forge_workspace=str(Path(spec.workspace).resolve()),
            artifacts=[patch_path],
            import_validation_modules=list(import_plan.modules),
            attempts=attempt,
        )


def generate_applyback_patch(
    spec: RewriteSpec,
    config: Config,
    request: ApplybackRequest,
    *,
    usage: UsageAccumulator | None = None,
) -> ApplybackResult:
    """Run bounded clean-room agent attempts and publish a validated patch."""
    base_commit = request.base_commit
    workspace = Path(spec.workspace).resolve()
    if not base_commit:
        return ApplybackResult(ok=False, error="apply-back requires an existing git base commit")
    if _git(workspace, "cat-file", "-e", f"{base_commit}^{{commit}}").returncode != 0:
        return ApplybackResult(ok=False, error=f"apply-back base commit is unavailable: {base_commit}")
    try:
        source_relative = Path(spec.source_kernel).resolve().relative_to(workspace).as_posix()
    except ValueError:
        return ApplybackResult(ok=False, error="source kernel is outside the framework workspace")
    flydsl_path = Path(spec.flydsl_kernel)
    if not flydsl_path.is_file():
        return ApplybackResult(ok=False, error="best FlyDSL kernel is unavailable for apply-back")
    if not DEFAULT_REWRITE_BUDGET.can_start_applyback(request.deadline_unix):
        return ApplybackResult(ok=False, error=_NO_TIME_FOR_APPLYBACK)
    resolved_framework = _infer_framework(spec)
    if resolved_framework not in protocol.SUPPORTED_FRAMEWORKS:
        return ApplybackResult(
            ok=False, error=f"apply-back framework could not be resolved to a supported value: {resolved_framework!r}"
        )

    reference_dir = Path(tempfile.mkdtemp(prefix="forge_rewrite_reference_"))
    reference_path = reference_dir / spec.flydsl_kernel_name
    reference_path.write_bytes(flydsl_path.read_bytes())
    brief = _IntegrationBrief(spec, resolved_framework, source_relative, reference_path)
    try:
        return _CleanRoomAttempts(config, request, brief, usage).run()
    finally:
        shutil.rmtree(reference_dir, ignore_errors=True)
