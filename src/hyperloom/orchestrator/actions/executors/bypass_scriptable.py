# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass scriptable (server-less) benchmark path."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import build_benchmark_env

from ._proposal_identity import is_executable, normalize_proposal


log = logging.getLogger(__name__)

#: The adapter's declared lever file, resolved inside ``$HYPERLOOM_BYPASS_SCRIPTS_DIR``.
#: The scripts dir is the adapter's contract directory and is already persisted and
#: re-exported on ``--resume``, so the declaration travels with the session.
DECLARED_LEVERS_FILENAME = "levers.json"


class DeclaredLeversError(ValueError):
    """A ``levers.json`` exists but cannot be used as a lever declaration."""


def declared_levers_path() -> Path | None:
    """Return the adapter's ``levers.json`` path, or ``None`` when unavailable."""
    scripts_dir = (os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR") or "").strip()
    if not scripts_dir:
        return None
    candidate = Path(scripts_dir) / DECLARED_LEVERS_FILENAME
    return candidate if candidate.is_file() else None


def load_declared_levers() -> list[dict[str, Any]]:
    """Load the lever grid a ``custom`` adapter declares for itself.

    A ``custom`` workload has no framework source for the model to read, so
    ``_default_grid_for_framework`` returns ``[]`` and EXPLORE has nothing to sweep.
    The adapter declares its own knob space instead.

    Each entry is judged by ``is_executable(normalize_proposal(...))`` -- the same
    rule, and the same owner, the executor applies when it builds the grid. A
    declaration that normalizes to nothing a restart could apply is rejected here
    rather than silently recorded as a baseline no-op later.

    Raises:
        DeclaredLeversError: When the file exists but is unreadable, is not valid
            JSON, is not a list of variant objects, or declares a variant no server
            restart could apply. Malformed input is loud by design: every silent
            path here surfaces later as ``empty_grid``, which reads as "the model
            proposed nothing".
    """
    path = declared_levers_path()
    if path is None:
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeclaredLeversError(f"{path} is unreadable: {exc}") from exc
    try:
        declared = json.loads(raw)
    except ValueError as exc:
        raise DeclaredLeversError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(declared, list):
        raise DeclaredLeversError(f"{path} must be a JSON list of variant objects, got {type(declared).__name__}")
    variants: list[dict[str, Any]] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict):
            raise DeclaredLeversError(f"{path}[{index}] must be a variant object, got {type(entry).__name__}")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise DeclaredLeversError(f"{path}[{index}] has no 'name'")
        if not is_executable(normalize_proposal(entry)):
            # Includes the shapes that *look* set but normalize away: an
            # ``extra_envs`` given as a string or a list becomes ``{}``.
            raise DeclaredLeversError(
                f"{path}[{index}] ({name!r}) declares nothing a restart could apply: "
                "set at least one of extra_args, extra_envs (an object), remove_args, "
                "unset_envs, or args_mode='replace'"
            )
        variant = dict(entry)
        # ``default_grid`` is the seed-grid provenance label; anything else is
        # attributed to the orchestration agent in the session breakdown.
        variant.setdefault("provenance", "default_grid")
        variants.append(variant)
    return variants


def declared_lever_payload(shared_state: Any, framework: str) -> list[dict[str, Any]]:
    """Return the ``custom`` adapter's declared levers the ledger has not benchmarked.

    Skips declarations whose name already appears in the explore ledger's ``tested``
    rows -- the bucket ``apply_explore_search_update`` actually writes. Re-seeding a
    name the ledger has measured re-spends the round's wall-clock budget ahead of the
    model's own proposals.
    """
    if framework != "custom":
        return []
    search = getattr(shared_state, "explore_search", None)
    tested = (search or {}).get("tested") if isinstance(search, dict) else None
    benched = {str(row.get("name") or "") for row in (tested or {}).values() if isinstance(row, dict)}
    fresh = [v for v in load_declared_levers() if str(v.get("name") or "") not in benched]
    if fresh:
        log.info(
            "explore: seeding %d declared lever variant(s) from %s",
            len(fresh),
            DECLARED_LEVERS_FILENAME,
        )
    return fresh


def _scriptable_script_name(framework: str, runner_type: str) -> str:
    """Return the scriptable entrypoint name (e.g. xdit_mi300x.sh)."""
    return f"{framework}_{runner_type}.sh"


def scriptable_script_candidates(
    framework: str,
    runner_type: str,
    inferencex_root: str,
    bench: dict[str, Any] | None = None,
) -> list[Path]:
    """Return the search list ``resolve_scriptable_script`` walks, in order."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    explicit = str((bench or {}).get("benchmark_script") or "").strip()
    if explicit:
        _add(Path(explicit))
    name = _scriptable_script_name(framework, runner_type)
    override = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    if override:
        _add(Path(override) / name)
    magpie_path = os.environ.get("MAGPIE_PATH", "").strip()
    if magpie_path:
        _add(Path(magpie_path, "Magpie", "scripts", "benchmark", name))
    _add(Path(inferencex_root, "benchmarks", name))
    return candidates


def resolve_scriptable_script(
    framework: str,
    runner_type: str,
    inferencex_root: str,
    bench: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve the scriptable benchmark script path."""
    for candidate in scriptable_script_candidates(framework, runner_type, inferencex_root, bench):
        if candidate.is_file():
            return candidate
    return None


def build_scriptable_env(
    bench: dict[str, Any],
    runner_type: str,
    workspace: Path,
    *,
    profile: bool = False,
    profile_dir: str | None = None,
) -> dict[str, str]:
    """Build the env for a scriptable benchmark script."""
    # Defaults are overridable by the YAML envs; run-scoped values are not.
    defaults: dict[str, str] = {"MODEL": str(bench.get("model") or os.environ.get("MODEL", ""))}
    if bench.get("precision"):
        defaults["PRECISION"] = str(bench["precision"])
    run_scoped: dict[str, str] = {
        "RUNNER_TYPE": runner_type,
        "RESULT_FILENAME": "inferencex_result",
        "RESULT_DIR": str(workspace),
    }
    # Scriptable scripts (e.g. xDiT) gate tracing on PROFILE=1 and read the trace dir from
    # VLLM/SGLANG_TORCH_PROFILER_DIR.
    if profile:
        run_scoped["PROFILE"] = "1"
        if profile_dir:
            run_scoped["VLLM_TORCH_PROFILER_DIR"] = profile_dir
            run_scoped["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    return build_benchmark_env(defaults, bench.get("envs"), run_scoped)


@dataclass(frozen=True)
class ScriptableBenchmark:
    """Which scriptable entrypoint runs, from which InferenceX checkout, with the benchmark config it reads."""

    framework: str
    runner_type: str
    inferencex_root: str
    bench: dict[str, Any]


def run_scriptable(
    benchmark: ScriptableBenchmark,
    workspace: Path,
    *,
    timeout_s: float,
    profile: bool = False,
    profile_dir: str | None = None,
) -> tuple[int, str | None]:
    """Run the scriptable benchmark script."""
    framework = benchmark.framework
    runner_type = benchmark.runner_type
    inferencex_root = benchmark.inferencex_root
    bench = benchmark.bench
    script = resolve_scriptable_script(framework, runner_type, inferencex_root, bench)
    if script is None:
        name = _scriptable_script_name(framework, runner_type)
        candidates = scriptable_script_candidates(framework, runner_type, inferencex_root, bench)
        tried = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        error = f"scriptable benchmark script not found for {name}"
        # Pre-spawn miss never opens Popen, so there is no child stderr.
        _write_stderr_log(workspace, f"{error}\ntried:\n{tried}\n")
        return 2, error
    env = build_scriptable_env(bench, runner_type, workspace, profile=profile, profile_dir=profile_dir)
    cmd = ["bash", str(script)]
    # Streamed straight to disk instead of captured in memory: a runner killed from outside (lease reap / OOM) must
    # still leave a forensic trail.
    with ExitStack() as stack:
        stdout_sink = stack.enter_context(_open_log_sink(workspace, "scriptable_stdout.log"))
        stderr_sink = stack.enter_context(_open_log_sink(workspace, "scriptable_stderr.log"))
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_sink,
            stderr=stderr_sink,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            _write_stderr_log(workspace, f"scriptable benchmark timed out after {timeout_s}s", append=True)
            return 124, None
    return proc.returncode, None


def _open_log_sink(workspace: Path, name: str):
    """Open a streaming log sink under ``workspace``, falling back to DEVNULL."""
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        return (workspace / name).open("wb")
    except OSError:
        return nullcontext(subprocess.DEVNULL)


def _write_stderr_log(workspace: Path, text: str, *, append: bool = False) -> None:
    """Persist a scriptable failure message to ``scriptable_stderr.log`` (best-effort)."""
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with (workspace / "scriptable_stderr.log").open("a" if append else "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass
