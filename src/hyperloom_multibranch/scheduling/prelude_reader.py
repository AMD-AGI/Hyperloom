# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one reader of the baseline-prelude child's session: the main flow's launch info, ``state.json`` and reports.

``state.json`` is the main flow's internal persistence format, not an interface, so every key read from it is read here
and nowhere else, and a schema version other than the pinned one is refused rather than guessed at. What the reader
returns is folded into the root :class:`~hyperloom.contracts.recipe.Recipe` by :func:`root_recipe`. The child runs with
enablement off and every optimization lever off, so nothing in it patches the framework or changes its launch: the root
is the framework checkout the operator named, at its ``HEAD``, launched as the main flow launches it. A child that
measured no baseline is refused, naming how it ended. The child's latest roofline profile is located on disk
(:attr:`PreludeArtifacts.profile`), not through ``state.json``: the main flow records its trace only once its own
analysis of it succeeded. The run that measured the child's anchoring baseline is located through ``state.json``
(:attr:`PreludeArtifacts.baseline_run`), so an AgentX session can take it as its baseline instead of measuring the root
again.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.jsonio import read_json_strict
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.recipe import ComponentPatches, Recipe, RecipeError
from hyperloom.inference_optimizer.breakdown.stop_reasons import outcome_status
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    detect_framework_version,
)

#: ``LATEST_STATE_SCHEMA_VERSION`` of src/hyperloom/orchestrator/state/shared_state.py, which this package may not
#: import; a test compares the two.
MB_LATEST_STATE_SCHEMA_VERSION = 6

#: How long one read-only git command may take before the checkout counts as unreadable.
GIT_TIMEOUT_SECONDS = 60.0

MB_STATE_FILE = "state.json"
#: The main flow seeds ``framework_repo_path`` from this variable of its environment
#: (src/hyperloom/inference_optimizer/cli/bootstrap.py), which the child inherits from the scheduler.
MB_FRAMEWORK_REPO_ENV = "FRAMEWORK_REPO_PATH"
#: The framework version the main flow records when its command line names none, before it detects one
#: (``resolve_framework_version`` of src/hyperloom/inference_optimizer/cli/bootstrap.py, which the scheduler may not
#: import: it loads the main flow's command line).
MB_FRAMEWORK_VERSION_ENV = "FRAMEWORK_VERSION"
#: The key of the main flow's ``--launch-info-file`` (src/hyperloom/inference_optimizer/cli ``_emit_launch_info``).
MB_LAUNCH_SESSION_DIR_KEY = "session_dir"
#: The decode ceiling of a roofline snapshot (src/hyperloom/inference_optimizer/roofline_snapshot.py).
MB_ROOFLINE_PEAK_KEY = "theoretical_peak_tok_per_sec"
#: Where the main flow's auto-roofline profile lands under its session directory, one directory per roofline run
#: (src/hyperloom/orchestrator/actions/executors/roofline.py), and the point config that run measured beside it.
MB_ROOFLINE_TRACE_GLOB = "runs/roofline/*/benchmark_*/torch_trace"
MB_POINT_CONFIG_FILE = "baseline_config.with_envs.yaml"
#: ``TRACE_GLOB`` of hyperloom_multibranch/toolbox/analysis.py, the trace a profile must hold for the toolbox to accept
#: it; importing that module would load the main flow's runtime, so a test compares the two.
MB_TRACE_FILE_GLOB = "*.trace.json*"
#: The ``outcome_status`` of a run that measured a baseline (src/hyperloom/inference_optimizer/breakdown/stop_reasons.py).
MB_OUTCOME_COMPLETED = "completed"
#: How the main flow marks the baseline attempt that became the anchor
#: (src/hyperloom/orchestrator/loop/_baseline_promotion.py), and the attempt statuses of one that measured and of one
#: that failed (src/hyperloom/orchestrator/state/shared_state.py ``record_action_attempt``).
MB_PROMOTED = "promoted"
MB_SUCCEEDED = "succeeded"
MB_FAILED = "failed"
DEFAULT_REMOTE = "origin"


class PreludeReadError(RuntimeError):
    """The child session is missing, of another schema, measured no baseline, or holds something the root recipe cannot
    express."""


@dataclass(frozen=True)
class UpstreamPin:
    """Where a component's source comes from: its git remote and the commit the patches apply on."""

    repo_url: str
    ref: str


@dataclass(frozen=True)
class ChildProfile:
    """The trace directory of one roofline profile of the child, the point config that run measured, and the run's id
    (its ``runs/roofline/<run_id>`` directory)."""

    run_id: str
    trace_dir: Path
    point_config: Path


@dataclass(frozen=True)
class ChildBaselineRun:
    """The child's run that measured its anchoring baseline of the root: ``run_dir`` is its ``runs/baseline/<id>``,
    ``benchmark_dir`` the benchmark directory below it that holds the report, ``point_config`` the point config it ran.
    ``wall_seconds`` is the run's time from launch to exit, boot included, and ``p50`` the P50 interactivity the child
    recorded for it."""

    run_dir: Path
    benchmark_dir: Path
    point_config: Path
    wall_seconds: float
    p50: float | None


@dataclass(frozen=True)
class PreludeArtifacts:
    """What a baseline prelude that measured its baseline left behind, as the root recipe and the session profile need
    it.

    ``framework_root`` is the framework checkout the operator named; it is ``None`` when the child named none or it is
    not a git checkout (an installed package). ``roofline_ceiling`` is keyed by the metric it bounds. ``profile`` is the
    child's latest roofline profile, ``None`` when it profiled nothing. ``baseline_run`` is the run of the child's
    anchoring baseline, ``None`` with the reason in ``baseline_run_refusal`` when it cannot stand for the root recipe:
    only an AgentX session needs it.
    """

    session_dir: Path
    framework_root: Path | None
    roofline_ceiling: Mapping[str, float]
    profile: ChildProfile | None
    baseline_run: ChildBaselineRun | None = None
    baseline_run_refusal: str = ""


def read_child_session(launch_info: Path) -> Path | None:
    """The child's session directory from its launch info; ``None`` when the child never wrote one."""

    if not launch_info.exists():
        return None
    document = _json_object(launch_info)
    session_dir = document.get(MB_LAUNCH_SESSION_DIR_KEY)
    if not isinstance(session_dir, str) or not session_dir:
        raise PreludeReadError(f"{launch_info} names no {MB_LAUNCH_SESSION_DIR_KEY}")
    return Path(session_dir)


def read_prelude(session_dir: Path) -> PreludeArtifacts:
    """Read the child session in ``session_dir``; raises :class:`PreludeReadError` on anything it cannot trust, and,
    before reading anything else, when the child measured no baseline, naming how it ended.

    Blocks: telling whether the framework root is a git checkout runs ``git``.
    """

    state = _json_object(session_dir / MB_STATE_FILE)
    version = state.get("schema_version")
    if version != MB_LATEST_STATE_SCHEMA_VERSION:
        raise PreludeReadError(
            f"{session_dir / MB_STATE_FILE} has schema version {version!r}; this reader knows "
            f"{MB_LATEST_STATE_SCHEMA_VERSION}"
        )
    stop_reason = str(state.get("stop_reason") or "")
    baseline_tput = _number(state.get("baseline_tput"), "baseline_tput")
    if outcome_status(stop_reason, baseline_tput) != MB_OUTCOME_COMPLETED:
        raise PreludeReadError(_no_baseline(state, stop_reason, baseline_tput))
    baseline_run, refusal = _baseline_run(state)
    return PreludeArtifacts(
        session_dir=session_dir,
        framework_root=_checkout(str(state.get("framework_repo_path") or "")),
        roofline_ceiling=_roofline_ceiling(state.get("roofline_snapshots")),
        profile=_latest_profile(session_dir),
        baseline_run=baseline_run,
        baseline_run_refusal=refusal,
    )


def _no_baseline(state: Mapping[str, object], stop_reason: str, baseline_tput: float) -> str:
    """How a child that measured no baseline ended: its stop reason, and the error class of its last failed baseline."""

    failed = [attempt for attempt in _baseline_attempts(state) if attempt.get("status") == MB_FAILED]
    cause = f"; its last baseline attempt failed with {failed[-1].get('error_class')!r}" if failed else ""
    return f"the child ended with stop_reason={stop_reason!r} and baseline_tput={baseline_tput}{cause}"


def _baseline_attempts(state: Mapping[str, object]) -> list[Mapping[str, object]]:
    attempts = state.get("baseline_attempts")
    return [attempt for attempt in (attempts if isinstance(attempts, list) else []) if isinstance(attempt, Mapping)]


def _baseline_run(state: Mapping[str, object]) -> tuple[ChildBaselineRun | None, str]:
    """The run of the child's anchoring baseline, or why there is none that measured the root recipe.

    The anchor is the last promoted baseline that succeeded, and it must be the one ``baseline_tput`` came from. A
    dropped measure round means the value is a cold start's.
    """

    promoted = [
        attempt
        for attempt in _baseline_attempts(state)
        if attempt.get("decision") == MB_PROMOTED and attempt.get("status") == MB_SUCCEEDED
    ]
    if not promoted:
        return None, "the child promoted no baseline that succeeded"
    anchor = promoted[-1]
    extras = _mapping(anchor.get("extras"), "baseline_attempts[].extras")
    if extras.get("baseline_tput") != state.get("baseline_tput"):
        return None, (
            f"the last promoted baseline measured {extras.get('baseline_tput')!r}, not the anchor's "
            f"baseline_tput {state.get('baseline_tput')!r}"
        )
    if state.get("baseline_measure_round_dropped"):
        return None, "the anchor's measure round was dropped, so its value is a cold start's"
    runtime = state.get("baseline_runtime_sec")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or runtime <= 0:
        return None, f"the child recorded no baseline_runtime_sec ({runtime!r})"
    workspace = anchor.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        return None, "the anchor names no workspace"
    benchmark_dir = Path(workspace)
    run_dir = benchmark_dir.parent
    perf = _mapping(state.get("baseline_perf"), "baseline_perf")
    p50 = perf.get(GRADED_INTVTY_P50)
    return ChildBaselineRun(
        run_dir=run_dir,
        benchmark_dir=benchmark_dir,
        point_config=run_dir / MB_POINT_CONFIG_FILE,
        wall_seconds=float(runtime),
        p50=float(p50) if isinstance(p50, (int, float)) and not isinstance(p50, bool) else None,
    ), ""


def _latest_profile(session_dir: Path) -> ChildProfile | None:
    """The roofline profile holding a trace whose directory was written last; ``None`` when the child left none.

    A failed capture leaves its trace directory empty, so a later failed run never hides an earlier profile.
    """

    traces = session_dir.glob(MB_ROOFLINE_TRACE_GLOB)
    written = [(mtime, trace) for trace in traces if (mtime := _trace_mtime(trace)) is not None]
    if not written:
        return None
    trace_dir = max(written)[1]
    run = trace_dir.parents[1]
    return ChildProfile(run_id=run.name, trace_dir=trace_dir, point_config=run / MB_POINT_CONFIG_FILE)


def _trace_mtime(trace_dir: Path) -> float | None:
    """When ``trace_dir`` was written, ``None`` when it holds no trace or cannot be read: locating the profile never
    fails reading the PRELUDE."""

    try:
        if next(trace_dir.rglob(MB_TRACE_FILE_GLOB), None) is None:
            return None
        return trace_dir.stat().st_mtime
    except OSError:
        return None


def launched_root(framework: str, env: Mapping[str, str], base_image_digest: str) -> Recipe:
    """The root recipe a child launched with ``env`` gives once it measured its baseline, known before it starts: the
    child runs on the checkout ``env`` names, as :func:`read_prelude` reads it back, and changes nothing of it. Raises
    :class:`PreludeReadError`. Blocks."""

    return root_recipe(framework, framework_pin(_checkout(env.get(MB_FRAMEWORK_REPO_ENV, ""))), base_image_digest)


def launched_framework_version(framework: str, env: Mapping[str, str]) -> str:
    """The version of ``framework`` a child launched with ``env`` records, as the main flow resolves it for a command
    line that names none: :data:`MB_FRAMEWORK_VERSION_ENV`, else the installed package's, ``""`` when that cannot be
    told. Blocks: detecting it imports the framework's package."""

    named = env.get(MB_FRAMEWORK_VERSION_ENV, "").strip()
    if named:
        return named
    detected = detect_framework_version(framework)
    return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected


def framework_pin(root: Path | None) -> UpstreamPin | None:
    """The framework checkout ``root``'s remote at its ``HEAD``, which nothing in the child patched; ``None`` without a
    checkout."""

    if root is None:
        return None
    remotes = _git(root, "remote").split()
    if not remotes:
        raise PreludeReadError(f"{root} has no git remote to pin the framework to")
    remote = DEFAULT_REMOTE if DEFAULT_REMOTE in remotes else remotes[0]
    return UpstreamPin(repo_url=_git(root, "remote", "get-url", remote), ref=_git(root, "rev-parse", "HEAD"))


def root_recipe(framework: str, pin: UpstreamPin | None, base_image_digest: str) -> Recipe:
    """The baseline as a recipe on ``base_image_digest``: the main flow's own launch, and ``framework`` at ``pin``
    without a patch."""

    try:
        components = () if pin is None else (ComponentPatches(framework, pin.repo_url, pin.ref),)
        return Recipe(base_image_digest=base_image_digest, components=components)
    except RecipeError as exc:
        raise PreludeReadError(f"the prelude's result is not a valid recipe: {exc}") from exc


def _json_object(path: Path) -> Mapping[str, object]:
    try:
        document = read_json_strict(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreludeReadError(f"cannot read {path}: {exc}") from exc
    return _mapping(document, str(path))


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PreludeReadError(f"{what} is not a JSON object: {value!r}")
    return value


def _number(value: object, what: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreludeReadError(f"{what} is not a number: {value!r}")
    return float(value)


def _roofline_ceiling(snapshots: object) -> Mapping[str, float]:
    """The decode ceiling of the latest snapshot that has one, as a bound on output throughput; empty when none has.

    The main flow keeps the newest snapshot last.
    """

    for snapshot in reversed(_snapshot_list(snapshots)):
        peak = snapshot.get(MB_ROOFLINE_PEAK_KEY) if isinstance(snapshot, Mapping) else None
        if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0:
            return {GRADED_OUTPUT: float(peak)}
    return {}


def _snapshot_list(snapshots: object) -> list[object]:
    if snapshots is None:
        return []
    if not isinstance(snapshots, list):
        raise PreludeReadError(f"roofline_snapshots is not a list: {snapshots!r}")
    return snapshots


def _checkout(path: str) -> Path | None:
    root = Path(path.strip()) if path.strip() else None
    return root if root is not None and _is_checkout(root) else None


def _is_checkout(root: Path) -> bool:
    try:
        _git(root, "rev-parse", "--show-toplevel")
    except PreludeReadError:
        return False
    return True


def _git(root: Path, *args: str) -> str:
    """The stripped standard output of ``git -C root args``. Blocks."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args), capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc}") from exc
    return completed.stdout.strip()


__all__ = [
    "MB_LATEST_STATE_SCHEMA_VERSION",
    "ChildBaselineRun",
    "ChildProfile",
    "PreludeArtifacts",
    "PreludeReadError",
    "UpstreamPin",
    "framework_pin",
    "launched_framework_version",
    "launched_root",
    "read_child_session",
    "read_prelude",
    "root_recipe",
]
