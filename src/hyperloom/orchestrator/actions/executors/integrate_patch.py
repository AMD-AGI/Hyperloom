# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply specialist patches to live framework roots and KEEP or REVERT by benchmark."""

from __future__ import annotations

import csv
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
    redact_secret_values,
)
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.gpu_types import amd_gpu_dispatch_identity
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...framework.paths import (
    resolve_session_framework_root,
    resolve_source_file_allowlist,
    resolved_within,
)
from ...specialists.patch_safety import (
    is_unified_diff,
    patch_escapes_tree,
    patch_targets_missing,
    resolve_patch_apply_root,
)
from ...state.shared_state import (
    inject_stack_base_params,
    resolve_anchor_with_drift,
    resolve_graded_comparison,
)
from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_UPSTREAM_PR
from hyperloom.common.env import is_truthy
from hyperloom.common.gain_math import gain_pct
from ..stop_attribution import stopped_by_the_run_class
from ...policy.gate import INTEGRATE_PATCH_PERMISSIVE_VERDICTS
from ._accuracy_gate import (
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
    accuracy_keep_block,
    accuracy_meets_floor,
    accuracy_passed,
    classify_accuracy_failure,
    eval_probe_summary,
    parse_eval_results,
    read_eval_probe,
)
from ._apply_feedback import ApplyFeedback, build_apply_feedback
from ._git import _run_git_cp
from ._patch_source_pr import (
    DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    _candidate_slug,
    materialize_candidate_patches,
)
from ._nogit_patch import (
    _P_LEVELS,
    _apply_patch_no_git,
    _is_git_tree,
    _is_within,
    _revert_patches_no_git,
)
from ._patch_snapshot import _git_commit_kept, _patch_touched_paths
from ._canonical_fingerprint import canonical_fingerprint
from ._grid_runner import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_VARIANT_TIMEOUT_SEC,
    GridVariant,
    VariantResult,
    _num_gpus_for_config,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from . import _framework_switch_manifest as _switch_manifest
from ._grid_server_args import compose_server_args
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


_HYPERLOOM_AUTO_STASH_MSG = "hyperloom-auto-stash: preserving user changes before candidate run"
# Deliberately shares no substring with the auto-stash tag: _find_hyperloom_auto_stash
# matches by message, and a quarantined merge must never be picked up and popped back.
_HYPERLOOM_QUARANTINE_STASH_MSG = "hyperloom-quarantine: unresolved merge cleared before candidate run"


# Enablement environment-setup replay: allowlist of install-only command shapes.
# A specialist may run arbitrary Bash in its own sandboxed session, but the
# durable *replay* performed here (before applying patches + booting) is limited
# to package/tool installation so a recorded ``setup_commands`` list can never be
# a vector for arbitrary side effects (rm, curl|bash, service restarts, etc.).
# Matched against the command with leading `sudo `/env-assignments stripped.
_SETUP_CMD_ALLOWLIST: tuple[str, ...] = (
    r"pip3?\s+install\b",
    r"(?:python3?|uv)\s+-m\s+pip\s+install\b",
    r"uv\s+pip\s+install\b",
    r"pip3?\s+uninstall\s+-y\b",
    # Creating an isolated environment to install INTO. Without these the only
    # spelling that survived the allowlist was installing into the system
    # interpreter (``PIP_BREAK_SYSTEM_PACKAGES=1 pip install``), so the gate was
    # steering repairs toward the less safe of the two options it had to choose
    # between. Creating a venv directory is bounded; breaking the system's
    # package manager is not.
    r"uv\s+venv\b",
    r"(?:python3?|uv)\s+-m\s+venv\b",
    r"apt(?:-get)?\s+(?:install|update)\b",
    r"npm\s+(?:install|i|ci)\b",
    r"npm\s+install\s+-g\b",
    r"pnpm\s+(?:install|add)\b",
    r"yarn\s+(?:add|install)\b",
    r"conda\s+install\b",
    r"mamba\s+install\b",
)
#: Directory prefixes whose basename may stand in for the whole path when the
#: allowlist is matched. Absolute and system-owned on purpose: the replay runs
#: the ORIGINAL command string, so anything a specialist can write to -- a
#: relative ``./pip``, a path under its own workspace -- must not be able to
#: borrow an allowlisted name. ``/opt/venv`` is the canonical ROCm stack this
#: repository installs into; the rest are the standard system bindirs.
_TRUSTED_BIN_PREFIX_RE = re.compile(r"^(?:/opt/[A-Za-z0-9._-]+|/usr(?:/local)?|/bin|/sbin)(?:/[A-Za-z0-9._-]+)*/")

#: Per-command clip in the rejection summary. Long enough to recognise the
#: command, short enough that twelve of them cannot bury the round's own reason.
_SKIPPED_CMD_CHARS = 160

_SETUP_CMD_MAX = 12  # cap on distinct setup commands per integrate
_SETUP_CMD_TIMEOUT_SEC = 1800  # 30 min per install command
# Two-sided band, in percent of the pre-patch base, that a switch-off parity leg
# must land inside. The rewrite workloads this gates measure with a run-to-run
# spread well under 1%, so a band this wide clears noise by a comfortable margin
# while still catching a patch that is not actually inert when disabled.
DEFAULT_SWITCH_OFF_PARITY_BAND_PCT = 2.0


#: Where the patches this action applies came from. Every source lands through
#: the same apply / vet / bench / KEEP-REVERT pipeline below; they differ only
#: in how the diff is obtained and, for ``upstream_pr``, in which gate admitted
#: it (see :meth:`IntegratePatchExecutor._stage_resolve`).
PATCH_SOURCE_SPECIALIST = "specialist_authored"
PATCH_SOURCE_UPSTREAM_PR = "upstream_pr"
PATCH_SOURCES = (PATCH_SOURCE_SPECIALIST, PATCH_SOURCE_UPSTREAM_PR)


def resolve_patch_source(params: Mapping[str, Any]) -> str:
    """Return the declared patch source, defaulting to the specialist lane.

    Args:
        params: Task params.

    Returns:
        One of :data:`PATCH_SOURCES`.
    """
    declared = str(params.get("patch_source") or "").strip().lower()
    return declared if declared in PATCH_SOURCES else PATCH_SOURCE_SPECIALIST


_LAUNCH_ONLY_MUTATION_FIELDS: tuple[str, ...] = (
    "patches",
    "enablement_base_patches",
    "localization_candidate",
    "runtime_candidate",
    "artifacts",
    "config_changes",
    "enablement_setup_commands",
)


def _parse_framework_switches(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read the framework-rewrite switch manifest for this integration.

    Looks in the task params first (an explicit dispatch), then in the
    specialist's done payload (the normal authoring path, possibly nested under
    ``payload``).

    Args:
        params: The integrate_patch task params.
        done_payload: The originating specialist's done payload, if any.

    Returns:
        ``(switches, problems)`` from :func:`_switch_manifest.parse_manifest`;
        ``([], [])`` when no manifest was delivered.
    """
    raw = params.get(_switch_manifest.MANIFEST_KEY)
    if not raw and isinstance(done_payload, dict):
        raw = done_payload.get(_switch_manifest.MANIFEST_KEY)
        if not raw:
            inner = done_payload.get("payload")
            if isinstance(inner, dict):
                raw = inner.get(_switch_manifest.MANIFEST_KEY)
    if not raw:
        return [], []
    # Env the benchmark already defines is reserved: a "switch" colliding with it
    # would be toggled by unrelated configuration rather than by the lever.
    reserved: set[str] = set()
    for source in (params.get("base_extra_envs"), params.get("extra_envs")):
        if isinstance(source, dict):
            reserved.update(str(k).strip().upper() for k in source)
    return _switch_manifest.parse_manifest(raw, reserved_env=reserved)


def _with_skipped_setup_reason(reason: str, setup_result: dict[str, Any]) -> str:
    """Append the allowlist rejections to a round's ``reason``.

    A rejected setup command was only ever a ``log.warning``. Downstream saw the
    round's outcome with no link to the cause, so the same authoring attempt was
    re-dispatched until the budget ran out -- each round proposing the same fix
    and each round having it silently dropped. Naming the rejection in the reason
    is what lets the next round (or an operator) see that the proposal was never
    the problem.

    Args:
        reason: The round's existing reason text.
        setup_result: The :func:`_run_setup_commands` result.

    Returns:
        ``reason`` unchanged when nothing was rejected, else ``reason`` with a
        one-line summary of the rejected commands appended.
    """
    # ``_run_setup_commands`` already stores the sanitised form, so for every
    # production caller this is a no-op. Applied again anyway: the lesson of the
    # gap this closes is that a safety step placed at the call sites protects
    # the call sites that exist, and the sanitiser is idempotent.
    skipped = [_sanitize_setup_command(c) for c in (setup_result.get("skipped") or []) if str(c).strip()]
    if not skipped:
        return reason
    listed = "; ".join(skipped[:_SETUP_CMD_MAX])
    if len(skipped) > _SETUP_CMD_MAX:
        listed += f"; (+{len(skipped) - _SETUP_CMD_MAX} more)"
    note = f"{len(skipped)} setup command(s) were REJECTED by the install-only allowlist and never ran: {listed}"
    return f"{reason} ({note})" if reason else note


def _sanitize_setup_command(cmd: str) -> str:
    """A rejected command in the form it is safe to store and hand back.

    Rejected commands are LLM-written text. They reach the journal, the report
    and the KB, and are read back into the next round's mandate, so a bearer
    token or a credentialed URL in one would outlive the round that produced it.
    Clipped as well, so a single rejected install naming a hundred packages
    cannot crowd out the reason it is reported alongside.
    """
    text = redact_secret_values(str(cmd).strip())
    return text if len(text) <= _SKIPPED_CMD_CHARS else text[:_SKIPPED_CMD_CHARS] + "..."


def _is_allowlisted_setup_command(cmd: str) -> bool:
    """True when ``cmd`` is an install-only command safe to replay.

    Strips a leading ``sudo``, any ``KEY=VALUE`` env-assignment prefixes and the
    executable's directory, then requires the remainder to start with a known
    package/tool installer. Rejects anything with shell control operators that
    could chain an arbitrary payload.
    """
    text = (cmd or "").strip()
    if not text:
        return False
    # Reject command substitution / backticks / newlines outright — these can
    # smuggle an arbitrary payload regardless of tokenization.
    if re.search(r"[`\n]|\$\(", text):
        return False
    # Guard against genuine shell chaining/redirection while allowing pip/pkg
    # version specifiers that legitimately contain ``>``/``<`` (e.g.
    # ``transformers>=4.58``). Neutralise the safe, non-shell uses first, then
    # reject any leftover metacharacter (the replay runs under ``shell=True``).
    scrubbed = text
    # Drop quoted segments (their contents cannot act as shell operators).
    scrubbed = re.sub(r"'[^']*'", " ", scrubbed)
    scrubbed = re.sub(r'"[^"]*"', " ", scrubbed)
    # Drop an unquoted pip-style version comparison only when it is attached to
    # the package token and the version starts with a digit (``pkg>=4.58``).
    # Whitespace-prefixed operators and non-version targets remain visible to
    # the metacharacter check below (``foo >evil``, ``2>evil``, ``foo <evil``).
    scrubbed = re.sub(r"(?<=[0-9A-Za-z_.\]])(?:>=|<=|>|<)(?=\d)", " ", scrubbed)
    # Any remaining shell chaining/redirection metacharacter => unsafe.
    if re.search(r"[;&|<>]", scrubbed):
        return False
    # Strip a leading sudo and leading KEY=VALUE env assignments.
    text = re.sub(r"^\s*sudo\s+", "", text)
    text = re.sub(r"^(?:\s*[A-Za-z_][A-Za-z0-9_]*=[^\s]*\s+)+", "", text)
    # Match on the executable's basename, but ONLY for an absolute path under a
    # system prefix. The patterns below are anchored, so without any
    # normalisation ``/opt/venv/bin/uv pip install X`` was REJECTED while
    # ``uv pip install X`` -- the same operation -- was allowed. Measured: two
    # sessions hit one missing dependency and got opposite outcomes, decided by
    # nothing but how the specialist happened to spell the path.
    #
    # The allowlist is checked against this normalised text, but
    # ``_run_setup_commands`` executes the ORIGINAL string under ``shell=True``.
    # So a blanket basename strip would let any binary in: ``./pip install foo``
    # normalises to an allowlisted ``pip install foo`` while running a script
    # the specialist just wrote into its own workspace. Restricting the strip to
    # absolute system prefixes keeps "which KIND of operation may replay" intact
    # -- the property line 105 promises -- while still treating a venv's own
    # interpreter as the interpreter it is.
    text = _TRUSTED_BIN_PREFIX_RE.sub("", text, count=1)
    return any(re.match(pat, text) for pat in _SETUP_CMD_ALLOWLIST)


_now_iso = functools.partial(now_iso, "auto")


def _resolve_setup_commands(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
) -> list[str]:
    """Resolve the ordered, deduped enablement setup commands to replay.

    Sources (in order; deduped preserving first occurrence): base commands
    stacked from prior rounds (``params['enablement_setup_commands']``) then the
    current specialist's ``specialist_done.setup_commands``. Non-string / blank
    entries are dropped; the list is capped at :data:`_SETUP_CMD_MAX`.

    Args:
        params: The integrate_patch task params.
        done_payload: The specialist ``specialist_done`` payload (may be None).

    Returns:
        list[str]: Ordered unique candidate setup commands (pre-allowlist).
    """
    out: list[str] = []
    seen: set[str] = set()
    sources: list[Any] = []
    base = params.get("enablement_setup_commands")
    if isinstance(base, list):
        sources.extend(base)
    if isinstance(done_payload, dict):
        dp = done_payload.get("setup_commands")
        if isinstance(dp, list):
            sources.extend(dp)
    for c in sources:
        s = str(c or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= _SETUP_CMD_MAX:
            break
    return out


def _run_setup_commands(commands: list[str], *, cwd: Path, log_dir: Path) -> dict[str, Any]:
    """Replay allowlisted enablement setup commands (installs) before boot.

    Runs each allowlisted command non-interactively with a per-command timeout,
    appending combined output to ``<log_dir>/enablement_setup.log``. Commands
    that fail the allowlist are skipped (never executed). A non-zero install is
    recorded but does NOT hard-fail the integration — the subsequent boot/gate
    is the source of truth for runnability.

    Args:
        commands: Candidate setup commands (already deduped / capped).
        cwd: Working directory for the commands.
        log_dir: Directory to write ``enablement_setup.log`` into.

    Returns:
        dict[str, Any]: ``{"applied": [...], "skipped": [...], "failed": [...]}``
        where ``applied`` are the allowlisted commands that ran (rc==0).
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    if not commands:
        return {"applied": applied, "skipped": skipped, "failed": failed}
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Logging is best-effort.
        pass
    log_path = log_dir / "enablement_setup.log"
    env = dict(os.environ)
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    for cmd in commands:
        if not _is_allowlisted_setup_command(cmd):
            # Sanitised HERE, not at the reporting sites. This list is copied
            # verbatim into every result payload that carries
            # ``setup_commands_skipped``, and a rejected command is LLM-written
            # text that can hold a bearer token or a credentialed URL. Doing it
            # at the four call sites protects those four; doing it at the source
            # protects the fifth as well.
            safe_cmd = _sanitize_setup_command(cmd)
            skipped.append(safe_cmd)
            # Also carried into the round's ``reason`` by
            # _with_skipped_setup_reason: a warning alone left the caller with an
            # outcome and no link to the cause, so the same proposal was
            # re-authored and re-dropped until the budget ran out. The log is a
            # disk-backed surface too, so it gets the sanitised form as well.
            log.warning("integrate_patch: skipping non-allowlisted enablement setup command: %s", safe_cmd)
            continue
        log.info("integrate_patch: enablement setup replay: %s", cmd)
        try:
            proc = subprocess.run(  # noqa: S602  # nosec B602 - allowlisted install-only shell command.
                cmd,
                shell=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=_SETUP_CMD_TIMEOUT_SEC,
            )
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"$ {cmd}\n{proc.stdout}\n{proc.stderr}\n(rc={proc.returncode})\n\n")
            except OSError:
                # Logging is best-effort.
                pass
            if proc.returncode == 0:
                applied.append(cmd)
            else:
                failed.append(cmd)
                log.warning("integrate_patch: enablement setup rc=%d for: %s", proc.returncode, cmd)
        except (subprocess.TimeoutExpired, OSError) as exc:
            failed.append(cmd)
            log.warning("integrate_patch: enablement setup errored (%s) for: %s", type(exc).__name__, cmd)
    return {"applied": applied, "skipped": skipped, "failed": failed}


def allowlisted_explicit_root(
    explicit: str,
    allowlist: tuple[str, ...] | None = None,
) -> Path | None:
    """Resolve a declared framework root, or ``None`` when it is not allowlisted.

    A root outside the allowlisted source scope is refused whatever its tree
    holds, so callers must ask this before blaming the patches for not
    matching it.

    Args:
        explicit: The declared ``framework_source_root``.
        allowlist: Pre-resolved allowlist, computed once by the caller when
            available, to avoid a redundant ``resolve_source_file_allowlist()``
            call.

    Returns:
        The resolved directory, or ``None`` when it is unreadable, absent, or
        outside the allowlisted source scope.
    """
    try:
        resolved = Path(explicit).resolve()
    except (OSError, RuntimeError):
        log.warning(
            "integrate_patch: framework_source_root override %r could not be resolved",
            explicit,
        )
        return None
    if not resolved.is_dir():
        log.warning(
            "integrate_patch: framework_source_root override %r does not exist",
            explicit,
        )
        return None
    effective_allowlist = allowlist if allowlist is not None else resolve_source_file_allowlist()
    if any(resolved_within(explicit, root) for root in effective_allowlist):
        return resolved
    log.warning(
        "integrate_patch: framework_source_root override %r rejected (outside trusted source scope)",
        explicit,
    )
    return None


def _read_patch_texts(patch_paths: list[Path] | None) -> list[str]:
    """Return the diff text of every patch that could be read.

    Args:
        patch_paths: Patch files to read.

    Returns:
        The texts that were readable; a shorter list than ``patch_paths`` means
        some patch is missing or unreadable.
    """
    texts: list[str] = []
    for patch in patch_paths or []:
        try:
            texts.append(patch.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


def _sole_patch_root(done_payload: dict[str, Any] | None) -> str | None:
    """Return the one apply root recorded for every patch, or ``None``.

    A set spanning two trees has no single apply root, so it falls back to
    resolution rather than silently picking one.

    Args:
        done_payload: The originating specialist's done payload, if any.

    Returns:
        The sole recorded root, or ``None``.
    """
    raw = (done_payload or {}).get("patch_roots")
    if not isinstance(raw, dict):
        return None
    roots = {str(v) for v in raw.values() if str(v).strip()}
    return roots.pop() if len(roots) == 1 else None


def _resolve_framework_root(
    explicit: str | None,
    patch_paths: list[Path] | None = None,
    patch_texts: list[str] | None = None,
    recorded_root: str | None = None,
) -> Path | None:
    """Pick one unambiguous framework root under the shared Patch rules.

    A ``recorded_root`` — carried from the authoring stage through
    ``done_payload["patch_roots"]`` — is authoritative and skips probing
    entirely. It is rejected outright when it falls outside the allowlist,
    exactly as a declared ``explicit`` root is.

    Without a recorded root, the decision falls through to
    :func:`~...specialists.patch_safety.resolve_patch_apply_root`. Without any
    patches to place, the session's declared root wins.

    Args:
        explicit: Declared framework root. Rejected when it resolves outside
            the trusted source scope.
        patch_paths: Patch files to place; unreadable ones are skipped.
        patch_texts: Patch diffs already in memory, placed alongside
            ``patch_paths``.
        recorded_root: The apply root recorded at authoring time, if any.

    Returns:
        The resolved root, or ``None`` when the patches name no single tree.
    """
    allowlist = resolve_source_file_allowlist()
    roots = [Path(root) for root in allowlist]

    if recorded_root:
        return allowlisted_explicit_root(recorded_root, allowlist=allowlist)

    explicit_path: Path | None = None
    if explicit:
        explicit_path = allowlisted_explicit_root(explicit, allowlist=allowlist)
        if explicit_path is None:
            return None

    texts = [str(text) for text in (patch_texts or []) if str(text).strip()]
    texts.extend(_read_patch_texts(patch_paths))
    has_patch_input = bool(patch_paths or patch_texts)
    if has_patch_input:
        session_root = resolve_session_framework_root()
        # The allowlist does not necessarily hold it: it discovers the unprefixed
        # env var, while the session root also answers to <FRAMEWORK>_REPO_PATH
        # and <FRAMEWORK>_DIR. Leaving it out turns the tree under optimisation
        # into a non-candidate, and default_root cannot stand in -- that is
        # consulted only for a create-only set, which has no pre-image to match.
        candidates = [Path(session_root), *roots] if session_root else list(roots)
        resolution = resolve_patch_apply_root(
            texts,
            explicit_root=explicit_path,
            candidate_roots=tuple(candidates),
            default_root=Path(session_root) if session_root else None,
        )
        if resolution.root is None:
            log.warning(
                "integrate_patch: Patch root resolution rejected: %s%s",
                resolution.reason,
                (f" matches={[str(root) for root in resolution.matches]!r}" if resolution.matches else ""),
            )
        return resolution.root

    if explicit_path is not None:
        return explicit_path
    session_root = resolve_session_framework_root()
    if session_root and Path(session_root).is_dir():
        return Path(session_root)
    for root in roots:
        if root.is_dir() and (root / ".git").exists():
            return root
    for root in roots:
        if root.is_dir():
            return root
    return None


def _run_git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    p_level: int,
    three_way: bool,
    check_only: bool,
) -> tuple[bool, str]:
    """Single ``git apply`` invocation at an explicit strip level.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        p_level: The ``-p<N>`` strip level.
        three_way: Whether to pass ``-3`` for a three-way merge.
        check_only: Whether to pass ``--check`` (dry run, no mutation).

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True on a zero return code.
    """
    args = ["-C", str(framework_root), "apply", f"-p{p_level}"]
    if three_way:
        args.append("-3")
    if check_only:
        args.append("--check")
    args.append(str(patch_path))
    cp = _run_git_cp(args, timeout=120.0)
    if cp is None:
        return False, "git apply spawn failed"
    return cp.returncode == 0, cp.stderr.strip()


def _derive_lane(params: dict[str, Any]) -> str:
    """Derive the retry lane name from integrate_patch params.

    Returns:
        ``"enablement"``, ``"perf_framework"``, or ``"perf_explore"``.
    """
    if params.get("enablement"):
        return "enablement"
    if params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"):
        return "perf_framework"
    return "perf_explore"


def _is_eval_origin(params: dict[str, Any]) -> bool:
    """Whether the enablement candidate came from the eval gate, not the boot gate."""
    return str(params.get("enablement_origin") or "") == "eval"


def _accuracy_delta_pct(measured: Any, baseline: Any) -> float | None:
    """Percent accuracy change of ``measured`` against ``baseline``.

    Returns ``None`` when either side is missing or the baseline is not
    positive, so callers fall back to their existing value.
    """
    try:
        m = float(measured)
        b = float(baseline)
    except (TypeError, ValueError):
        return None
    if b <= 0.0:
        return None
    return (m - b) / b * 100.0


def _preflight_missing_targets(
    framework_root: Path,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Return per-patch records for patches whose modify/delete targets are
    absent from ``framework_root`` at every ``-p`` strip level.

    A hallucinated-layout patch (e.g. modifying a CUDA-only file on a ROCm
    build) can never apply; flagging it here yields an actionable advisory
    instead of an opaque ``git_apply_failed`` after a wasted apply attempt.
    Patches supplied directly via ``params.patches`` bypass the
    authoring-time ``specialists.patch_safety`` vetting gate
    (:func:`vet_patches`), so they are checked here.

    Args:
        framework_root: The git checkout the patches target.
        patch_paths: The patch files to preflight.

    Returns:
        A list of per-patch records (``patch`` + ``missing_targets``) for
        patches whose targets are absent at every strip level.
    """
    records: list[dict[str, Any]] = []
    for patch in patch_paths:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        missing = patch_targets_missing(text, framework_root)
        if missing:
            records.append({"patch": str(patch), "missing_targets": missing})
    return records


def _localization_paths_outside_allowlist(
    touched_paths: list[str],
    framework_root: Path | None,
    allow_roots: list[str],
) -> list[str]:
    """Return the touched paths that resolve outside the allowed source roots.

    A localization diff may only write under the source-file allowlist or the
    attempt-local root. Paths are resolved against ``framework_root`` when
    relative. Returns the offending paths (empty when all are in-bounds).
    Fail closed: with no trusted write root, every non-empty touched path is
    treated as out of bounds.
    """
    roots = [Path(r).resolve() for r in allow_roots if str(r).strip()]
    if framework_root is not None:
        roots.append(Path(framework_root).resolve())
    if not roots:
        return [str(rel or "").strip() for rel in touched_paths if str(rel or "").strip()]
    outside: list[str] = []
    for rel in touched_paths:
        rel_s = str(rel or "").strip()
        if not rel_s:
            continue
        base = framework_root if framework_root is not None else Path("/")
        cand = (base / rel_s).resolve() if not Path(rel_s).is_absolute() else Path(rel_s).resolve()
        if not any(_is_within(cand, root) for root in roots):
            outside.append(rel_s)
    return outside


def _detect_p_level(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool,
) -> int | None:
    """Return the first ``-p`` level whose ``--check`` applies cleanly.

    Args:
        framework_root: The git checkout to test against.
        patch_path: The patch file to probe.
        three_way: Whether to probe with ``-3``.

    Returns:
        The first ``-p<N>`` level that applies cleanly, or ``None`` when none
        do.
    """
    for lvl in _P_LEVELS:
        ok, _ = _run_git_apply(
            framework_root,
            patch_path,
            p_level=lvl,
            three_way=three_way,
            check_only=True,
        )
        if ok:
            return lvl
    return None


def _git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool = False,
    check_only: bool = False,
) -> tuple[bool, str]:
    """Run ``git apply [-3] -p<auto> [--check] <patch>`` inside
    ``framework_root``, auto-detecting the strip level. Returns
    ``(ok, stderr)``.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        three_way: Whether to pass ``-3`` for a three-way merge.
        check_only: Whether to only check (dry run) rather than mutate.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the apply (or check)
        succeeds.
    """
    lvl = _detect_p_level(framework_root, patch_path, three_way=three_way)
    if lvl is None:
        # Surface a representative error at the git-native default level.
        return _run_git_apply(
            framework_root,
            patch_path,
            p_level=1,
            three_way=three_way,
            check_only=check_only,
        )
    if check_only:
        return True, ""
    return _run_git_apply(
        framework_root,
        patch_path,
        p_level=lvl,
        three_way=three_way,
        check_only=False,
    )


def _git_reverse_applies_cleanly(framework_root: Path, patch_path: Path) -> bool:
    """True when ``patch_path`` is already fully applied in ``framework_root``.

    The git-channel twin of
    :func:`._nogit_patch._reverse_applies_cleanly`. ``git apply -R --check``
    succeeds only when every hunk's *post*-state is already present, i.e. the
    tree already equals what a forward apply would produce. Read-only:
    ``--check`` never mutates the tree.

    Args:
        framework_root: The git checkout the patch targets.
        patch_path: The patch file to probe.

    Returns:
        ``True`` when some strip level reverse-checks cleanly.
    """
    from ._nogit_patch import _P_LEVELS

    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is not None and cp.returncode == 0:
            return True
    return False


def _git_apply_collect_feedback(
    framework_root: Path,
    patch_path: Path,
    *,
    three_way: bool = False,
) -> "tuple[bool, str, ApplyFeedback | None]":
    """Like :func:`_git_apply` but also returns an :class:`ApplyFeedback` on failure.

    On success returns ``(True, "", None)``.  On failure returns
    ``(False, stderr, ApplyFeedback)`` where *ApplyFeedback* carries the
    combined stderr from both the initial and ``-3`` attempt, the list of
    tried ``-p`` levels, and a source-context snippet.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        three_way: Whether to fall back to ``-3`` on first failure.

    Returns:
        ``(ok, err, feedback)`` — feedback is ``None`` on success.
    """
    from ._nogit_patch import _P_LEVELS

    # Collect per-level check stderr for the feedback record.
    tried_levels: list[int] = []
    level_stderrs: list[str] = []
    for lvl in _P_LEVELS:
        ok_check, stderr_check = _run_git_apply(
            framework_root, patch_path, p_level=lvl, three_way=three_way, check_only=True
        )
        tried_levels.append(lvl)
        if stderr_check:
            level_stderrs.append(f"-p{lvl}: {stderr_check}")
        if ok_check:
            # Level works; now apply for real.
            ok_apply, stderr_apply = _run_git_apply(
                framework_root, patch_path, p_level=lvl, three_way=three_way, check_only=False
            )
            if ok_apply:
                return True, "", None
            feedback = build_apply_feedback(
                patch_path,
                channel="git",
                tried_levels=tried_levels,
                stderr=stderr_apply,
                framework_root=framework_root,
            )
            return False, stderr_apply, feedback

    # All levels failed; retry with -3.
    if not three_way:
        ok3, err3, fb3 = _git_apply_collect_feedback(framework_root, patch_path, three_way=True)
        if ok3:
            return True, "", None
        # Still nothing. Distinguish "does not apply" from "already applied":
        # a specialist often writes both a superset patch and the subset it
        # contains, so applying one leaves the other a satisfied no-op that
        # ``git apply --check`` nonetheless rejects. A clean *reverse* check
        # succeeds only when every hunk's post-state is already in the tree,
        # which is exactly what a forward apply would have produced -- so treat
        # it as success rather than failing the whole combo. Partial overlap
        # fails the reverse check and stays a real failure.
        if _git_reverse_applies_cleanly(framework_root, patch_path):
            log.info(
                "integrate_patch: %s is already fully applied (clean git apply -R --check); treating as a no-op",
                patch_path.name,
            )
            return True, "", None
        # Merge both sets of stderrs.
        all_stderrs = "\n".join(level_stderrs)
        if err3:
            all_stderrs = all_stderrs + "\n-3 retry: " + err3 if all_stderrs else "-3 retry: " + err3
        feedback = build_apply_feedback(
            patch_path,
            channel="git",
            tried_levels=tried_levels,
            stderr=all_stderrs,
            framework_root=framework_root,
        )
        return False, all_stderrs, feedback

    all_stderrs = "\n".join(level_stderrs)
    feedback = build_apply_feedback(
        patch_path,
        channel="git",
        tried_levels=tried_levels,
        stderr=all_stderrs,
        framework_root=framework_root,
    )
    return False, all_stderrs, feedback


def _git_apply_reverse(
    framework_root: Path,
    patch_path: Path,
) -> tuple[bool, str]:
    """Reverse-apply ``patch_path`` (``git apply -R -p<auto>``) as the REVERT
    path; caller falls back to ``git checkout`` on failure. Auto-detects the
    same strip level the forward apply used via ``-R --check``.

    Args:
        framework_root: The git checkout to reverse-apply into.
        patch_path: The patch file to reverse-apply.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the reverse apply
        succeeds.
    """
    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is None:
            return False, "git apply -R spawn failed"
        if cp.returncode != 0:
            continue
        cp2 = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", str(patch_path)],
            timeout=120.0,
        )
        if cp2 is None:
            return False, "git apply -R spawn failed"
        if cp2.returncode == 0:
            return True, ""
        return False, cp2.stderr.strip()
    return False, f"git apply -R: no matching -p level for {patch_path}"


def _find_hyperloom_auto_stash(framework_root: Path) -> str:
    """Return the newest Hyperloom auto-stash ref, or ``""`` if absent."""
    cp = _run_git_cp(
        ["-C", str(framework_root), "stash", "list", "--format=%gd:%gs"],
        timeout=30.0,
    )
    if cp is None:
        return ""
    if cp.returncode != 0:
        return ""
    for line in cp.stdout.splitlines():
        ref, _sep, msg = line.partition(":")
        if ref and _HYPERLOOM_AUTO_STASH_MSG in msg:
            return ref
    return ""


def _git_quarantine_unmerged(framework_root: Path, paths: list[str]) -> tuple[bool, str]:
    """Bank unresolved-merge paths in a stash entry that is never popped back.

    ``git stash`` refuses to run at all while the index carries unmerged
    entries, so they are staged first — staging is what marks a conflict
    resolved — and the working-tree content, conflict markers and all, is what
    gets banked. Recover it with ``git stash list | grep hyperloom-quarantine``.

    Emptying the index is not the same as ending the merge: ``MERGE_HEAD``
    outlives both the staging and the stash, and while it stands the next
    ``git commit`` is a *merge* commit. A KEEP would then carry a second parent
    and silently claim the whole of the other side as accepted work, which is
    the one thing "every KEEP is a commit, so HEAD is the accepted stack"
    cannot survive. ``git merge --quit`` forgets the merge without touching the
    index or the working tree.

    Args:
        framework_root (Path): The framework repo.
        paths (list[str]): Repo-relative paths in an unresolved merge state.

    Returns:
        tuple[bool, str]: ``(ok, error)``; ``error`` is empty on success.
    """
    add = _run_git_cp(["-C", str(framework_root), "add", "--", *paths], timeout=60.0)
    if add is None:
        return False, "git add failed"
    if add.returncode != 0:
        return False, f"git add rc={add.returncode}: {add.stderr.strip()}"
    cp = _run_git_cp(
        [
            "-C",
            str(framework_root),
            "stash",
            "push",
            "-m",
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "--",
            *paths,
        ],
        timeout=60.0,
    )
    if cp is None:
        return False, "git stash push failed"
    if cp.returncode != 0:
        return False, f"git stash push rc={cp.returncode}: {cp.stderr.strip()}"
    quit_cp = _run_git_cp(["-C", str(framework_root), "merge", "--quit"], timeout=30.0)
    if quit_cp is None or quit_cp.returncode != 0:
        # Nothing was banked that a later revert cannot reach, but leaving
        # MERGE_HEAD standing would mislabel the next KEEP, so refuse rather
        # than proceed on a repo that is still mid-merge.
        detail = "git merge --quit failed" if quit_cp is None else quit_cp.stderr.strip()
        return False, f"merge state could not be cleared: {detail}"
    return True, ""


def _git_unmerged_paths(framework_root: Path) -> list[str]:
    """Return repo-relative paths left in an unresolved merge state, if any."""
    cp = _run_git_cp(["-C", str(framework_root), "ls-files", "-u", "--full-name"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        return []
    seen: list[str] = []
    for line in (cp.stdout or "").splitlines():
        _meta, _sep, path = line.partition("\t")
        path = path.strip()
        if path and path not in seen:
            seen.append(path)
    return seen


def _git_restore_to_head(framework_root: Path, paths: list[str] | None = None) -> tuple[bool, str]:
    """Force ``paths`` (or the whole tree) back to HEAD, clearing any merge state.

    Every KEEP is committed, so HEAD is the accepted stack: restoring to it drops
    candidate work and keeps everything the loop has accepted.

    Args:
        framework_root (Path): The git checkout to restore.
        paths (list[str] | None): Repo-relative paths, or ``None`` for the tree.

    Returns:
        tuple[bool, str]: ``(ok, stderr)``.
    """
    target = paths if paths else ["."]
    # `checkout --force HEAD --` resolves an unmerged index, which plain
    # `checkout -- .` refuses to touch.
    cp = _run_git_cp(
        ["-C", str(framework_root), "checkout", "--force", "HEAD", "--", *target],
        timeout=60.0,
    )
    if cp is None:
        return False, "git checkout spawn failed"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    return True, ""


def _git_stash_if_dirty(framework_root: Path) -> tuple[str, str]:
    """Stash uncommitted user changes so destructive resets don't lose them.

    Only stashes when the working tree is dirty (``git status --porcelain``
    is non-empty). The stash message is tagged for easy retrieval via
    ``git stash list | grep hyperloom-auto-stash``.

    Returns:
        ``(state, note)`` where ``state`` is one of:

        - ``"clean"`` — working tree was already clean, safe to proceed.
        - ``"stashed"`` — dirty tree was successfully stashed; ``note`` is the
          stash ref to restore when the candidate finishes.
        - ``"failed"`` — tree is dirty but stash command failed; callers
          MUST NOT proceed with destructive operations.
    """
    cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
    if cp is None:
        return "failed", "git status check failed"
    if cp.returncode != 0:
        # Non-git directory or other git status errors: treat as clean.
        log.debug(
            "integrate_patch: git status rc=%d in %s (not a git repo?), treating as clean",
            cp.returncode,
            framework_root,
        )
        return "clean", ""
    if not cp.stdout.strip():
        return "clean", ""
    # git refuses to stash while an unresolved merge stands, so every later
    # candidate would abort here forever. Clear it — but bank the content
    # instead of overwriting it. Usually this is wreckage from an earlier cycle;
    # nothing here can prove that, and a merge the operator started themselves
    # is not ours to throw away. Quarantine keeps both cases recoverable at the
    # cost of one stash entry. It is deliberately not the auto-stash tag, so it
    # is never popped back: restoring conflict markers into the source is what
    # made every benchmark fail to parse the model.
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_quarantine_unmerged(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s had %d path(s) in an unresolved merge (%s); moved to a '%s' stash entry%s",
            framework_root,
            len(unmerged),
            ", ".join(unmerged[:5]),
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "" if ok else f" FAILED: {err}",
        )
        if not ok:
            return "failed", f"unresolved merge could not be quarantined: {err}"
        cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
        if cp is None:
            return "failed", "git status check failed"
        if not cp.stdout.strip():
            return "clean", ""
    cp2 = _run_git_cp(
        ["-C", str(framework_root), "stash", "push", "-u", "-m", _HYPERLOOM_AUTO_STASH_MSG],
        timeout=60.0,
    )
    if cp2 is None:
        return "failed", "git stash push failed"
    if cp2.returncode == 0:
        stash_ref = _find_hyperloom_auto_stash(framework_root) or "stash@{0}"
        log.info(
            "integrate_patch: stashed user changes in %s as %s",
            framework_root,
            stash_ref,
        )
        return "stashed", stash_ref
    return "failed", f"git stash push rc={cp2.returncode}: {cp2.stderr.strip()}"


def _git_restore_stash_if_needed(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
) -> str:
    """Restore the user-change stash created before candidate mutation."""
    if stash_state != "stashed":
        return ""
    ref = stash_ref or _find_hyperloom_auto_stash(framework_root)
    if not ref:
        return "auto-stash ref not found; user changes remain in git stash"
    cp = _run_git_cp(["-C", str(framework_root), "stash", "pop", "--index", ref], timeout=120.0)
    if cp is None:
        return f"git stash pop failed; user changes remain in {ref}"
    if cp.returncode == 0:
        log.info("integrate_patch: restored user changes from %s", ref)
        return ""
    # A pop that cannot merge leaves conflict markers in source. Left there they
    # break every later measurement silently -- the benchmark fails to parse the
    # file, and git will not stash again while the merge is unresolved. Put the
    # tree back at HEAD instead; the stash is deliberately NOT dropped, so the
    # work is still there under `git stash list`.
    detail = (cp.stderr or "").strip()
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_restore_to_head(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s did not merge back into %s; restored %d path(s) to HEAD and kept the stash%s",
            ref,
            framework_root,
            len(unmerged),
            "" if ok else f" (restore FAILED: {err})",
        )
        if not ok:
            detail = f"{detail}; tree left conflicted: {err}"
    return f"git stash pop {ref} rc={cp.returncode}: {detail}; user changes remain in git stash"


def _restore_stash_logged(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
) -> str:
    """Restore a pre-candidate stash, reporting a failure to do so.

    Args:
        framework_root (Path): The tree the stash was taken from.
        stash_state (str): The state :func:`_git_stash_if_dirty` returned.
        stash_ref (str): The stash ref to pop.

    Returns:
        str: The failure note, or ``""`` when nothing was left in the stash.
    """
    note = _git_restore_stash_if_needed(framework_root, stash_state, stash_ref)
    if note:
        log.warning("integrate_patch: user-change stash restore failed: %s", note)
    return note


def _with_stash_restore(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Restore a pre-candidate stash before returning an executor result."""
    note = _restore_stash_logged(framework_root, stash_state, stash_ref)
    if not note:
        return result
    out = dict(result)
    out["stash_restore_error"] = note
    return out


def _git_checkout_clean(framework_root: Path) -> tuple[bool, str]:
    """Restore the working tree to HEAD and remove untracked candidate files.

    This is the REVERT: HEAD is the accepted stack because every KEEP is
    committed, so what this discards is exactly the candidate. User changes must
    already have been stashed before the candidate was applied.

    Args:
        framework_root (Path): Directory to run ``git checkout`` in.

    Returns:
        tuple[bool, str]: ``(ok, stderr)`` where ``ok`` is ``True`` on
        return code 0.
    """
    ok, err = _git_restore_to_head(framework_root)
    if not ok:
        return False, err
    cp2 = _run_git_cp(["-C", str(framework_root), "clean", "-fd"], timeout=60.0)
    if cp2 is None:
        return False, "git clean spawn failed"
    return cp2.returncode == 0, cp2.stderr.strip()


def _resolve_patch_paths(
    *,
    specialist_workspace: Path,
    explicit_patches: list[str] | None,
    done_payload: dict[str, Any] | None,
) -> list[Path]:
    """Resolve the list of patch files to apply.

    Order: ``params.patches`` → ``specialist_done.patches_written`` →
    filesystem scan of ``specialist_workspace/{worktree/,}patches/``.
    Entries normalised to absolute Paths; missing ones logged + dropped.

    Security: a resolved patch path must live inside the specialist workspace
    (or its worktree); an absolute path pointing outside the sandbox is dropped.
    Both sides are ``resolve()``-d first so a symlinked workspace still matches.

    Args:
        specialist_workspace: The specialist task workspace to resolve
            relative paths / scan for patches.
        explicit_patches: Explicit patch paths from params, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        The resolved, existing patch files as absolute Paths.
    """
    candidates: list[str] = []
    if explicit_patches:
        candidates.extend(str(p) for p in explicit_patches)
    elif done_payload and isinstance(done_payload.get("patches_written"), list):
        candidates.extend(str(p) for p in done_payload["patches_written"] if p)
    else:
        for base in (
            specialist_workspace / "worktree" / "patches",
            specialist_workspace / "patches",
        ):
            if base.is_dir():
                for p in sorted(base.glob("*.patch")):
                    candidates.append(str(p))
                for p in sorted(base.glob("*.diff")):
                    candidates.append(str(p))

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]

    out: list[Path] = []
    for c in candidates:
        p = Path(c)
        # Resolve relative paths against the specialist workspace + worktree.
        if not p.is_absolute():
            for base in (
                specialist_workspace / "worktree",
                specialist_workspace,
            ):
                cand = base / c
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            log.warning(
                "integrate_patch: patch %r not found (specialist_workspace=%s)",
                c,
                specialist_workspace,
            )
            continue
        resolved = p.resolve()
        if not any(_is_within(resolved, root) for root in allowed_roots):
            log.warning(
                "integrate_patch: patch %r resolves outside the specialist workspace (%s); dropping for safety",
                c,
                specialist_workspace,
            )
            continue
        out.append(resolved)
    return out


@dataclass
class _ArtifactSpec:
    """One resolved non-diff tuned artifact to install at integration.

    Attributes:
        source: Absolute path to the artifact file inside the specialist
            workspace / worktree (sandbox-validated).
        target: Absolute install path inside an allowlisted framework root
            (sandbox-validated; no escape).
        rel_target: The framework-relative target, normalized to the matched
            allowlisted root via ``_resolve_artifact_target`` (an author's
            absolute target is converted to this relative form). Used for
            reporting AND as the framework-relative key for the durable KEEP
            source snapshot.
        root: The allowlisted root ``rel_target`` is relative to. The KEEP
            source snapshot is keyed on one root, so an artifact installed into
            a different tree than the patches must be recognisable as such.
        kind: Free-form artifact kind label (e.g. ``config_json``).
        description: Free-form human description.
    """

    source: Path
    target: Path
    rel_target: str
    root: Path
    kind: str = ""
    description: str = ""


def _resolve_artifact_target(rel_target: str) -> tuple[Path, str, Path] | None:
    """Resolve an artifact target (framework-relative, or absolute) to a path.

    A relative target picks the allowlisted framework root whose tree already
    contains the target's parent directory (so a ``vllm/...`` config lands under
    the vllm root); else the first existing root. An absolute target is accepted
    ONLY when it resolves strictly inside an allowlisted root. Either way the
    resolved path must stay within the chosen root (no ``..`` escape).

    Args:
        rel_target: The install path authored by the specialist (framework-
            relative, or an absolute path inside an allowlisted root).

    Returns:
        A ``(absolute_target, framework_relative_target, root)`` tuple, or
        ``None`` when nothing resolves safely. Callers MUST persist the relative
        target AND the root so the durable KEEP source snapshot captures the
        installed file even when the author used an absolute path.
    """
    rel = (rel_target or "").strip()
    if not rel or ".." in Path(rel).parts:
        return None
    roots = [Path(r).resolve() for r in resolve_source_file_allowlist()]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    # An absolute target is accepted only when it resolves strictly inside an
    # allowlisted framework root.
    if Path(rel).is_absolute():
        cand = Path(rel).resolve()
        for root in roots:
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix(), root
        return None
    # Prefer a root whose tree already holds the target's parent dir.
    for root in roots:
        cand = (root / rel).resolve()
        if not _is_within(cand, root):
            continue
        if cand.parent.is_dir():
            return cand, cand.relative_to(root).as_posix(), root
    # Fall back to the first root that keeps the path contained.
    for root in roots:
        cand = (root / rel).resolve()
        if _is_within(cand, root):
            return cand, cand.relative_to(root).as_posix(), root
    return None


def _resolve_artifact_specs(
    *,
    specialist_workspace: Path,
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> tuple[list[_ArtifactSpec], list[dict[str, str]]]:
    """Resolve non-diff tuned artifacts to install.

    Order: ``params.artifacts`` → ``specialist_done.artifacts_written``. Each
    entry is ``{source, target, kind, description}``: ``source`` is resolved
    inside the specialist workspace/worktree (sandbox) and ``target`` is
    resolved inside an allowlisted framework root. Malformed / out-of-sandbox
    entries are dropped and reported.

    Args:
        specialist_workspace: The specialist task workspace.
        explicit_artifacts: ``params.artifacts`` override list, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        A ``(specs, errors)`` tuple: resolved specs, plus per-entry error
        records (``{artifact, error}``) for entries that could not be resolved.
    """
    raw: list[Any] = []
    if explicit_artifacts:
        raw = list(explicit_artifacts)
    elif done_payload and isinstance(done_payload.get("artifacts_written"), list):
        raw = list(done_payload["artifacts_written"])

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]
    specs: list[_ArtifactSpec] = []
    errors: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            errors.append({"artifact": str(entry), "error": "not_a_mapping"})
            continue
        src_rel = str(entry.get("source") or "").strip()
        tgt_rel = str(entry.get("target") or "").strip()
        if not src_rel or not tgt_rel:
            errors.append({"artifact": json.dumps(entry), "error": "missing_source_or_target"})
            continue
        # Resolve source inside the workspace sandbox.
        src = Path(src_rel)
        if not src.is_absolute():
            for base in (specialist_workspace / "worktree", specialist_workspace):
                cand = base / src_rel
                if cand.exists():
                    src = cand
                    break
        if not src.exists() or not src.resolve().is_file():
            errors.append({"artifact": src_rel, "error": "source_not_found"})
            continue
        src_resolved = src.resolve()
        if not any(_is_within(src_resolved, root) for root in allowed_roots):
            errors.append({"artifact": src_rel, "error": "source_outside_workspace"})
            continue
        resolved = _resolve_artifact_target(tgt_rel)
        if resolved is None:
            errors.append({"artifact": tgt_rel, "error": "target_unresolved_or_escapes_root"})
            continue
        target, rel_norm, root = resolved
        specs.append(
            _ArtifactSpec(
                source=src_resolved,
                target=target,
                rel_target=rel_norm,
                root=root,
                kind=str(entry.get("kind") or "").strip(),
                description=str(entry.get("description") or "").strip(),
            )
        )
    return specs, errors


def _is_aiter_gemm_model_config(spec: _ArtifactSpec) -> bool:
    """Return whether an artifact is an AITER runtime GEMM model-config CSV."""
    if spec.source.suffix.lower() != ".csv":
        return False
    kind = spec.kind.strip().lower()
    target = spec.target.as_posix().lower()
    filename = spec.target.name.lower()
    is_aiter_model_config = "/aiter/configs/model_configs/" in target
    return is_aiter_model_config and (kind == "model_config" or "_tuned_gemm" in filename)


def _validate_aiter_gemm_artifacts(
    specs: list[_ArtifactSpec],
    *,
    gpu_type: str | None,
) -> list[dict[str, str]]:
    """Reject AITER GEMM CSVs that cannot dispatch on the target GPU.

    A model-config seed containing rows for another architecture, or placeholder
    rows that still require an offline tuning step, has no runtime effect. Such
    an artifact must not enter E2E promotion because benchmark variance could
    otherwise attribute an unrelated gain to it.
    """
    identity = amd_gpu_dispatch_identity(gpu_type)
    if identity is None:
        return []
    expected_gfx, expected_cu_num = identity
    errors: list[dict[str, str]] = []
    required_columns = {"gfx", "cu_num", "us", "kernelName"}

    for spec in specs:
        if not _is_aiter_gemm_model_config(spec):
            continue
        base_error = {
            "artifact": str(spec.source),
            "expected_gfx": expected_gfx,
            "expected_cu_num": str(expected_cu_num),
        }
        try:
            with spec.source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                if not required_columns.issubset(fieldnames):
                    errors.append(
                        {
                            **base_error,
                            "error": "invalid_aiter_gemm_schema",
                        }
                    )
                    continue
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            errors.append({**base_error, "error": "invalid_aiter_gemm_csv"})
            continue

        def _cu_num_matches(row: dict[str, Any]) -> bool:
            raw = str(row.get("cu_num") or "").strip()
            if not raw:
                return False
            try:
                return int(float(raw)) == expected_cu_num
            except ValueError:
                return False

        target_rows = [
            row for row in rows if str(row.get("gfx") or "").strip().lower() == expected_gfx and _cu_num_matches(row)
        ]
        if not target_rows:
            errors.append({**base_error, "error": "no_target_gpu_rows"})
            continue

        invalid_target_rows = 0
        for row in target_rows:
            kernel_name = str(row.get("kernelName") or "").strip().lower()
            try:
                runtime_us = float(str(row.get("us") or "0").strip())
            except ValueError:
                runtime_us = 0.0
            if not math.isfinite(runtime_us) or runtime_us <= 0.0 or "placeholder" in kernel_name:
                invalid_target_rows += 1
        if invalid_target_rows:
            errors.append(
                {
                    **base_error,
                    "error": "target_gpu_rows_not_runtime_ready",
                    "invalid_rows": str(invalid_target_rows),
                    "target_rows": str(len(target_rows)),
                }
            )
    return errors


def _read_done_payload(workspace: Path) -> dict[str, Any] | None:
    """Read and parse ``specialist_done.json`` from a workspace.

    Args:
        workspace (Path): The specialist task workspace directory.

    Returns:
        dict[str, Any] | None: The parsed payload, or ``None`` when the
        file is absent or cannot be parsed.
    """
    done = workspace / "specialist_done.json"
    if not done.exists():
        return None
    try:
        return json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "integrate_patch: failed to parse %s: %r",
            done,
            exc,
        )
        return None


_FRAMEWORK_KB_PROVENANCE_PREFIX = "specialist:serving:framework"


def _stamp_framework_kb_provenance(
    done_payload: dict[str, Any] | None,
    *,
    params: dict[str, Any],
    shared_state: Any,
) -> None:
    """Ensure a FRAMEWORK-dispatched deliverable carries KB-writeback provenance.

    Stamps the ``specialist:serving:framework...`` provenance prefix (that
    :meth:`IntegratePatchExecutor._find_frameworkoposal` requires) from the
    dispatch context, so same-framework deliverables reach ``lessons.jsonl``.

    Mutates ``done_payload["proposal_set"][0]`` in place; no-ops when this
    action was not dispatched from FRAMEWORK authoring, or when a proposal
    already carries a matching provenance.

    Args:
        done_payload: The specialist's parsed ``specialist_done.json`` (may
            be ``None``/malformed; no-ops in that case).
        params: The ``integrate_patch`` action's dispatch params (carries
            ``framework_agent_authoring`` / ``framework_agent_candidate_id``
            when this run came from FRAMEWORK_AGENT).
        shared_state: The run's ``SharedState`` (best-effort ``framework``
            read for the provenance suffix).
    """
    if not isinstance(done_payload, dict):
        return
    if not params.get("framework_agent_authoring"):
        return
    pr_url = str(params.get("framework_agent_candidate_id") or "").strip()
    if not pr_url:
        return
    proposals = done_payload.get("proposal_set")
    if not isinstance(proposals, list) or not proposals or not isinstance(proposals[0], dict):
        # No proposal_set entry to stamp; synthesize a minimal anchor.
        done_payload["proposal_set"] = [{}]
        proposals = done_payload["proposal_set"]
    target = proposals[0]
    existing = str(target.get("provenance") or "")
    if existing.startswith(_FRAMEWORK_KB_PROVENANCE_PREFIX):
        return  # cross-framework (or already-stamped) path already complies
    framework = str(getattr(shared_state, "framework", "") or "").strip().lower()
    target["provenance"] = (
        f"{_FRAMEWORK_KB_PROVENANCE_PREFIX}:{framework}" if framework else _FRAMEWORK_KB_PROVENANCE_PREFIX
    )
    target.setdefault("fa_pr_url", pr_url)
    target.setdefault("framework", framework)


def _enforce_critic_gate(
    shared_state: Any,
    subject: str,
) -> "dict[str, Any] | None":
    """Enforce a permissive Critic verdict on ``subject`` before any side
    effect; returns a ``rejected_by_critic`` dict on failure, else ``None``
    when no SharedState is available or the verdict is permissive."""
    if shared_state is None:
        return None
    try:
        recorded = shared_state.get_specialist_patch_verdict(subject)
    except AttributeError:
        recorded = ""
    if (recorded or "").lower() not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
        _detail = f"verdict {recorded!r}" if recorded else "no Critic verdict on record"
        # No side effect has occurred yet; reject cleanly (nothing to revert).
        return {
            "status": "rejected_by_critic",
            "specialist_task_id": subject,
            "patches_applied": [],
            "patches_reverted": [],
            "config_changes_applied": {},
            "reason": (
                f"integrate_patch requires a permissive Critic verdict "
                f"(approve/advise) for {subject!r}; {_detail}. Refusing to run."
            ),
        }
    return None


class IntegratePatchExecutor:
    """ActionRunner for the ``integrate_patch`` action (PR-A4)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the integrate-patch executor.

        Args:
            session_dir (Path | str | None): Session output directory;
                auto-resolved when ``None``.
            default_config_path (Path | str | None): Fallback benchmark
                config path, if any.
            variant_timeout_sec (int): Per-variant benchmark hard timeout.
                Defaults to :data:`DEFAULT_VARIANT_TIMEOUT_SEC`.
            keep_threshold_pct (float): Minimum gain to KEEP a patch.
                Defaults to :data:`DEFAULT_KEEP_THRESHOLD_PCT`.
        """
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        # Both are set per round by _stage_apply and tell the revert the tree was
        # written even when no patch landed in ``applied``.
        self._apply_attempted: bool = False
        self._ip_base_artifact_replayed = False

    async def __call__(self, ctx) -> dict[str, Any]:
        """Apply a specialist's patches/config changes and benchmark them."""
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}

        early = await self._stage_resolve(ctx, params, extra)
        if early is not None:
            return early

        # _stage_resolve populates these onto ctx for stage communication.
        specialist_task_id: str = ctx._ip_specialist_task_id  # type: ignore[attr-defined]
        shared_state = ctx._ip_shared_state  # type: ignore[attr-defined]
        done_payload: dict[str, Any] = ctx._ip_done_payload  # type: ignore[attr-defined]

        # Provision an attempt-scoped runtime AFTER the Critic gate (in
        # _stage_resolve) and BEFORE any patch apply / setup replay.
        provision_early = await self._stage_provision_attempt_runtime(ctx, params, specialist_task_id)
        if provision_early is not None:
            return provision_early

        # Localize a merged-PR / vendored closure into the source tree. Fetch
        # happens post-Critic; a compiled/build closure defers to a clean
        # revert. Localized patches are prepended in _stage_apply.
        localize_early = await self._stage_localize_source(ctx, params, specialist_task_id)
        if localize_early is not None:
            return localize_early

        # The apply and the gate both mutate the framework tree behind the
        # operator's auto-stash, and both cross awaits while it is on the stack --
        # the apply stage writes a KB record on each of its failure verdicts. So
        # the guard spans both: whichever stage was running, the candidate is
        # taken back out and the stash handed back, and the stop is re-raised
        # rather than graded. Each stage publishes its tree-mutation bookkeeping
        # to ``ctx`` as it becomes real, because that is all the handler can see
        # when the stop arrives mid-stage.
        try:
            apply_result = await self._stage_apply(ctx, params, extra, specialist_task_id, shared_state, done_payload)
            if apply_result is not None:
                return apply_result

            # _stage_apply populates these.
            output_root: Path = ctx._ip_output_root  # type: ignore[attr-defined]
            framework_root: Path | None = ctx._ip_framework_root  # type: ignore[attr-defined]
            stash_state: str = ctx._ip_stash_state  # type: ignore[attr-defined]
            stash_note: str = ctx._ip_stash_note  # type: ignore[attr-defined]
            applied: list[Path] = ctx._ip_applied  # type: ignore[attr-defined]
            applied_artifacts: list[dict[str, Any]] = ctx._ip_applied_artifacts  # type: ignore[attr-defined]
            config_changes_applied: dict[str, str] = ctx._ip_config_changes_applied  # type: ignore[attr-defined]
            extra_server_args_applied: str = ctx._ip_extra_server_args_applied  # type: ignore[attr-defined]
            extra_envs_applied: dict[str, str] = ctx._ip_extra_envs_applied  # type: ignore[attr-defined]
            dropped_env_overrides: list[str] = ctx._ip_dropped_env_overrides  # type: ignore[attr-defined]
            setup_result: dict[str, Any] = ctx._ip_setup_result  # type: ignore[attr-defined]

            return await self._stage_gate(
                ctx,
                params,
                extra,
                specialist_task_id=specialist_task_id,
                shared_state=shared_state,
                done_payload=done_payload,
                output_root=output_root,
                framework_root=framework_root,
                stash_state=stash_state,
                stash_note=stash_note,
                applied=applied,
                applied_artifacts=applied_artifacts,
                config_changes_applied=config_changes_applied,
                extra_server_args_applied=extra_server_args_applied,
                extra_envs_applied=extra_envs_applied,
                dropped_env_overrides=dropped_env_overrides,
                setup_result=setup_result,
            )
        except BaseException:
            self._undo_ungraded_candidate(ctx)
            raise

    # ---------------------------------------------------------------------------
    # Stage helpers (called sequentially by __call__)
    # ---------------------------------------------------------------------------

    async def _stage_resolve(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Guards (multi-node, task-id, workspace, Critic) + param normalisation.

        Returns an early-exit result dict on failure, or None to continue.
        Stores resolved values as ``ctx._ip_*`` attributes for the next stages.
        """
        from ._multi_node_env import is_multi_node

        if is_multi_node():
            return {
                "status": "skipped",
                "skipped_reason": "multi_node_unsupported",
                "specialist_task_id": str(params.get("specialist_task_id") or "").strip(),
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "reason": (
                    "specialist integrate_patch is not supported in "
                    "multi-node mode (no git-diff pod fan-out); skipped "
                    "without applying any patch. Other actions "
                    "(baseline/profile/explore/sweep/roofline) continue "
                    "normally. Use the kernel-agent integrate path (which "
                    "fans out via `multi_node apply-patch`) or run single-node."
                ),
            }

        shared_state = extra.get("shared_state") or extra.get("state")
        if shared_state is not None and not params.get("accuracy_baseline"):
            _base_acc = getattr(shared_state, "baseline_accuracy", 0.0)
            if isinstance(_base_acc, (int, float)) and _base_acc > 0:
                params["accuracy_baseline"] = float(_base_acc)

        # Launch-only mode: pure bench of a pre-built runtime (no specialist, no Critic).
        if params.get("enablement_launch_only"):
            mutation_fields = [key for key in _LAUNCH_ONLY_MUTATION_FIELDS if params.get(key)]
            if mutation_fields:
                return {
                    "status": "failed",
                    "error_class": "launch_only_mutation_forbidden",
                    "error": (
                        "enablement_launch_only accepts a pre-built runtime_override "
                        f"but no mutation fields; received {mutation_fields}"
                    ),
                    "patches_applied": [],
                    "patches_reverted": [],
                    "config_changes_applied": {},
                }
            task_id = ctx.task.task_id
            scratch = runs_dir(self.session_dir, "integrate_patch", task_id)
            scratch.mkdir(parents=True, exist_ok=True)
            ctx._ip_specialist_task_id = task_id  # type: ignore[attr-defined]
            ctx._ip_shared_state = shared_state  # type: ignore[attr-defined]
            ctx._ip_specialist_workspace = scratch  # type: ignore[attr-defined]
            ctx._ip_done_payload = {}  # type: ignore[attr-defined]
            return None

        # Upstream-PR mode has no specialist to look up and no specialist
        # verdict to enforce: the Critic verdict on this proposal is the gate.
        if resolve_patch_source(params) == PATCH_SOURCE_UPSTREAM_PR:
            early = self._stage_resolve_upstream_pr(ctx, params, shared_state)
            if early is not None:
                return early
            return None

        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "integrate_patch requires params.specialist_task_id "
                    "(the completed specialist whose worktree carries "
                    "the patches to integrate)"
                ),
            }
        # Rebind from live current_best so a task queued before an Explore KEEP
        # still measures against the real stack top.
        if shared_state is not None:
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)
        # Specialist workspace conventionally at runs/specialist/<id>/.
        specialist_workspace = runs_dir(self.session_dir, "specialist", specialist_task_id)
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (f"specialist workspace not found at {specialist_workspace}"),
                "specialist_task_id": specialist_task_id,
            }

        # Critic-verdict gate — enforced BEFORE any side effect (setup replay,
        # stash, patch/artifact apply, pod fan-out). Paths that bypass PolicyGate
        # (notably a queued/resume-dispatched task) are not re-validated there, so
        # a forged coordinator.db row with no genuine Critic verdict must be
        # rejected here, all-or-nothing, before it can install packages or mutate
        # the live framework tree. specialist_patch_verdicts is a Coordinator-only
        # CORE_STATE_FIELD an LLM/forged row cannot write, and a legitimate
        # integrate_patch always has its verdict persisted before the queued task
        # is created (see intent_router._handle_single_verdict), so a genuine
        # task is unaffected. No-op when SharedState is absent.
        critic_reject = _enforce_critic_gate(shared_state, specialist_task_id)
        if critic_reject is not None:
            return critic_reject

        done_payload = _read_done_payload(specialist_workspace)
        _stamp_framework_kb_provenance(done_payload, params=params, shared_state=shared_state)

        ctx._ip_specialist_task_id = specialist_task_id  # type: ignore[attr-defined]
        ctx._ip_shared_state = shared_state  # type: ignore[attr-defined]
        ctx._ip_specialist_workspace = specialist_workspace  # type: ignore[attr-defined]
        ctx._ip_done_payload = done_payload  # type: ignore[attr-defined]
        return None

    def _stage_resolve_upstream_pr(
        self,
        ctx,
        params: dict[str, Any],
        shared_state: Any,
    ) -> dict[str, Any] | None:
        """Resolve an upstream-PR candidate into patches on the task's own scratch.

        Sets ``params['patches']`` so the shared apply stage below sees the same
        explicit-paths input a specialist lane produces, and points the
        workspace at this task's scratch dir since there is no specialist one.

        Args:
            ctx: The runner context; stage state is published onto it.
            params: Task params, mutated with the resolved patch paths.
            shared_state: SharedState, or ``None``.

        Returns:
            A terminal result when the candidate has no permissive Critic
            verdict or could not be materialised, else ``None``.
        """
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": "patch_source=upstream_pr requires params.candidate (the discovered PR row)",
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }
        if shared_state is not None:
            # Same rebind the specialist lane does: a task queued before a KEEP
            # landed must still measure against the real stack top.
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)

        # A queued or resume-dispatched task is not re-validated by PolicyGate,
        # and this lane fetches a remote diff into the live framework tree.
        critic_reject = _enforce_critic_gate(
            shared_state,
            str(params.get("framework_agent_candidate_id") or "").strip(),
        )
        if critic_reject is not None:
            return critic_reject

        task_id = ctx.task.task_id
        scratch = runs_dir(self.session_dir, "integrate_patch", task_id)
        scratch.mkdir(parents=True, exist_ok=True)
        framework_root = _resolve_framework_root(str(params.get("framework_source_root") or "").strip() or None)
        if framework_root is None:
            return {
                "status": "failed",
                "error_class": "framework_root_unresolved",
                "error": "cannot resolve a framework source root to apply the candidate to",
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }

        materialized = materialize_candidate_patches(
            candidate=candidate,
            params=params,
            framework_root=framework_root,
            output_root=scratch,
            slug=_candidate_slug(candidate),
            diff_fetch_timeout_sec=DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
        )
        if materialized.failure is not None:
            return {
                **materialized.failure,
                "candidate": candidate,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "patch_source_mode": materialized.mode,
                "workspace": str(scratch),
            }
        params["patches"] = [str(p) for p in materialized.patches]
        params["patch_source_mode"] = materialized.mode

        ctx._ip_specialist_task_id = task_id  # type: ignore[attr-defined]
        ctx._ip_shared_state = shared_state  # type: ignore[attr-defined]
        ctx._ip_specialist_workspace = scratch  # type: ignore[attr-defined]
        ctx._ip_done_payload = {}  # type: ignore[attr-defined]
        return None

    async def _stage_provision_attempt_runtime(
        self,
        ctx: Any,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Provision the attempt-scoped runtime from ``params['runtime_candidate']``.

        No-op when no candidate is present or in multi-node mode.
        Runs a disk preflight, delegates provision+probe to the framework
        adapter, and on success stores the resolved runtime on
        ``ctx._ip_provision_result`` / ``ctx._ip_stack_action`` for the gate to
        activate via the YAML-layer ``runtime_override``. Returns an early-exit
        ``reverted`` dict on any provision failure (no patch side effects yet),
        or ``None`` to continue.
        """
        ctx._ip_provision_result = None  # type: ignore[attr-defined]
        ctx._ip_stack_action = None  # type: ignore[attr-defined]
        raw = params.get("runtime_candidate")
        if not isinstance(raw, dict) or not raw:
            return None

        from ._multi_node_env import is_multi_node

        if is_multi_node():
            log.info("integrate_patch: skipping runtime provision in multi-node mode")
            return None

        from ...framework.adapters import get_adapter
        from ...framework.stack_actions import EnablementStackAction

        action = EnablementStackAction.from_state(raw)
        attempt_dir = (
            self.session_dir
            / "enablement"
            / "stacks"
            / (action.framework or "unknown")
            / (specialist_task_id or "attempt")
        )

        from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

        try:
            disk_preflight(attempt_dir.parent, n_candidates=1)
        except DiskPreflightError as exc:
            return {
                "status": "reverted",
                "error_class": "disk_preflight_failed",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision aborted: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — preflight is best-effort advisory
            log.warning("integrate_patch: disk preflight raised (%r); continuing", exc)

        adapter = get_adapter(action.framework)
        try:
            result = adapter.provision(action, attempt_dir)
            if result.ok and not adapter.probe(result, action):
                from ...framework.stack_actions import ProvisionResult as _PR

                result = _PR(ok=False, log_path=result.log_path, error="adapter probe failed after provision")
        except Exception as exc:  # noqa: BLE001 — provision failure is a clean revert, not a crash
            log.exception("integrate_patch: attempt-runtime provision raised")
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision raised: {exc!r}",
            }

        if not result.ok:
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_failed",
                "error": result.error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": f"attempt-runtime provision failed: {result.error}",
                "provision_result": result.to_state(),
            }

        # Record the attempt venv root on the action so KEEP can persist it and
        # resume/GC can find it.
        action = EnablementStackAction.from_state({**action.to_state(), "attempt_venv_root": result.runtime.venv_root})
        ctx._ip_provision_result = result  # type: ignore[attr-defined]
        ctx._ip_stack_action = action  # type: ignore[attr-defined]
        ctx._ip_attempt_venv_root = result.runtime.venv_root  # type: ignore[attr-defined]
        log.info(
            "integrate_patch: attempt runtime provisioned for %s (venv=%s, versions=%s)",
            action.framework,
            result.runtime.venv_root,
            result.installed_versions,
        )
        return None

    @staticmethod
    def _gc_attempt_dir(attempt_dir: Path) -> None:
        """Remove a half/failed attempt-runtime dir (best-effort)."""
        try:
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001 — GC is best-effort
            log.debug("integrate_patch: attempt-dir GC failed for %s", attempt_dir, exc_info=True)

    async def _stage_localize_source(
        self,
        ctx: Any,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Fetch/synthesize a localization diff and stage it for _stage_apply.

        No-op when no ``localization_candidate`` is present or in multi-node
        mode. Fetches the merged-PR / vendored diff (post-Critic), rejects a
        compiled / build-backend closure to a clean revert, enforces the
        source-file allowlist (+ the attempt-local root only), and
        writes the diff to a patch file recorded on ``ctx._ip_localization_patches``
        which ``_stage_apply`` prepends to the patch set. Returns an early-exit
        ``reverted`` dict on any gate/fetch failure (no tree mutation yet), or
        ``None`` to continue.
        """
        ctx._ip_localization_patches = []  # type: ignore[attr-defined]
        ctx._ip_localization_manifest = {}  # type: ignore[attr-defined]
        raw = params.get("localization_candidate")
        if not isinstance(raw, dict) or not raw:
            return None

        from ._multi_node_env import is_multi_node

        if is_multi_node():
            log.info("integrate_patch: skipping localization in multi-node mode")
            return None

        from ...framework.localization import build_localization_diff
        from ...framework.stack_actions import EnablementStackAction

        action = EnablementStackAction.from_state(raw)

        from hyperloom.agents.framework.sources import github as _gh

        def _base_reverted(error_class: str, reason: str) -> dict[str, Any]:
            return {
                "status": "reverted",
                "error_class": error_class,
                "error": reason,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "enablement": True,
                "reason": reason,
            }

        try:
            diff_text, touched_paths, verdict = build_localization_diff(
                action,
                fetch_pr_patches=lambda slug, num: _gh.pr_patches(slug, num),
                fetch_raw_file=lambda slug, ref, path: _gh.fetch_raw_file(slug, ref, path),
            )
        except Exception as exc:  # noqa: BLE001 — fetch failure is a clean revert
            log.exception("integrate_patch: localization fetch raised")
            return _base_reverted("localization_fetch_failed", f"localization fetch raised: {exc!r}")

        if not verdict.is_localizable:
            error_class = (
                "localization_rung5_deferred" if verdict.kind == "needs_rung5" else "localization_fetch_failed"
            )
            return _base_reverted(error_class, f"localization not applicable: {verdict.reason}")
        if not diff_text.strip():
            return _base_reverted("localization_fetch_failed", "localization produced an empty diff")

        # Allowlist gate: touched paths must resolve under the source-file
        # allowlist or the attempt-local root only (no global env mutation).
        framework_root: Path | None = _resolve_framework_root(
            params.get("framework_source_root") or None, patch_paths=[]
        )
        allow_roots = list(resolve_source_file_allowlist())
        attempt_root = str(getattr(ctx, "_ip_attempt_venv_root", "") or "")
        if attempt_root:
            allow_roots.append(str(Path(attempt_root).parent))
        outside = _localization_paths_outside_allowlist(touched_paths, framework_root, allow_roots)
        if outside:
            return _base_reverted(
                "localization_outside_allowlist",
                f"localization touches path(s) outside the allowlist: {outside[:8]}",
            )

        loc_dir = runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id)
        loc_dir = loc_dir / "localization"
        loc_dir.mkdir(parents=True, exist_ok=True)
        gap_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.gap_id or "localization")
        patch_path = loc_dir / f"{gap_slug}.patch"
        patch_path.write_text(diff_text, encoding="utf-8")

        ctx._ip_localization_patches = [patch_path]  # type: ignore[attr-defined]
        ctx._ip_stack_action = action  # type: ignore[attr-defined]
        ctx._ip_localization_touched = list(touched_paths)  # type: ignore[attr-defined]
        log.info(
            "integrate_patch: localization staged %s (%d file(s), kind=%s)",
            patch_path,
            len(touched_paths),
            action.kind,
        )
        return None

    @staticmethod
    def _publish_gate_state(
        ctx: Any,
        *,
        output_root: Path,
        config_changes_applied: dict[str, str],
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        dropped_env_overrides: list[str],
        setup_result: dict[str, Any],
    ) -> None:
        """Publish the values ``_stage_run`` reads back after ``_stage_apply``.

        Every field is required and keyword-only, so an exit that omits one
        fails here rather than in the gate. Tree-mutation state is published
        separately, as the tree takes it.
        """
        ctx._ip_output_root = output_root  # type: ignore[attr-defined]
        ctx._ip_config_changes_applied = config_changes_applied  # type: ignore[attr-defined]
        ctx._ip_extra_server_args_applied = extra_server_args_applied  # type: ignore[attr-defined]
        ctx._ip_extra_envs_applied = extra_envs_applied  # type: ignore[attr-defined]
        ctx._ip_dropped_env_overrides = dropped_env_overrides  # type: ignore[attr-defined]
        ctx._ip_setup_result = setup_result  # type: ignore[attr-defined]

    async def _stage_apply(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Setup replay, patch/artifact apply, pending_integrate sentinel.

        Returns an early-exit result dict on failure/no-patches/apply_only,
        or None to continue to bench+gate. Stores output values as ``ctx._ip_*``.
        """
        self._ip_base_artifact_replayed = False
        setup_result: dict[str, Any] = {"applied": [], "skipped": [], "failed": []}
        if bool(params.get("enablement")):
            setup_cmds = _resolve_setup_commands(params=params, done_payload=done_payload)
            if setup_cmds:
                setup_result = _run_setup_commands(
                    setup_cmds,
                    cwd=self.session_dir,
                    log_dir=runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id),
                )

        specialist_workspace: Path = ctx._ip_specialist_workspace  # type: ignore[attr-defined]
        explicit_patches = params.get("patches") or None
        patch_paths = _resolve_patch_paths(
            specialist_workspace=specialist_workspace,
            explicit_patches=(list(explicit_patches) if isinstance(explicit_patches, list) else None),
            done_payload=done_payload,
        )
        # Prepend localized closure patches (applied first, before this round's
        # patch) so the enablement composes on top of the localization.
        localization_patches = list(getattr(ctx, "_ip_localization_patches", None) or [])
        if localization_patches:
            seen_loc = {str(p) for p in patch_paths}
            prefix_loc = [p for p in localization_patches if p.is_file() and str(p) not in seen_loc]
            if prefix_loc:
                log.info("integrate_patch: prepending %d localization patch(es)", len(prefix_loc))
                patch_paths = prefix_loc + list(patch_paths)
        base_patches = params.get("enablement_base_patches")
        if bool(params.get("enablement")) and isinstance(base_patches, list) and base_patches:
            seen = {str(p) for p in patch_paths}
            prefix: list[Path] = []
            for bp in base_patches:
                bp_path = Path(str(bp))
                if bp_path.is_file() and str(bp_path) not in seen:
                    prefix.append(bp_path)
                    seen.add(str(bp_path))
            if prefix:
                log.info(
                    "integrate_patch: enablement stacking %d base patch(es) before this round's patch",
                    len(prefix),
                )
                patch_paths = prefix + list(patch_paths)

        config_changes = dict(params.get("config_changes") or {})
        if not config_changes and done_payload:
            cc = done_payload.get("config_changes")
            if isinstance(cc, dict):
                config_changes = {str(k): str(v) for k, v in cc.items()}
        proposal_extra_args_raw = params.get("extra_server_args")
        proposal_extra_args = (
            proposal_extra_args_raw
            if isinstance(proposal_extra_args_raw, str) and proposal_extra_args_raw.strip()
            else ""
        )
        proposal_extra_envs = dict(config_changes)
        raw_extra_envs = params.get("extra_envs")
        if isinstance(raw_extra_envs, dict):
            proposal_extra_envs.update({str(k): str(v) for k, v in raw_extra_envs.items()})
        proposal_extra_envs, _dropped = filter_untrusted_env_mapping(
            proposal_extra_envs,
            allow_predicate=is_allowed_variant_env_key,
        )
        dropped_env_overrides = sorted(_dropped)
        if dropped_env_overrides:
            log.warning(
                "integrate_patch: dropping unsafe env override keys: %s",
                ", ".join(dropped_env_overrides),
            )

        # Framework-rewrite switches. Every rewrite in such a patch sits behind
        # a switch that defaults OFF, so the applied patch is inert and benching
        # it as-is would measure the baseline. Turn the switches on for the
        # measurement and carry the parsed manifest to the gate, which needs the
        # dependency edges to decide between a throughput KEEP and an inert one.
        switch_manifest, switch_problems = _parse_framework_switches(
            params=params,
            done_payload=done_payload,
        )
        undeclared_gates = _switch_manifest.undeclared_switch_gates(patch_paths, switch_manifest)
        if undeclared_gates:
            reason = (
                f"patch gates on undeclared environment switch(es) "
                f"{', '.join(undeclared_gates)}: every gate a framework rewrite "
                f"introduces must be declared in the '{_switch_manifest.MANIFEST_KEY}' "
                f"manifest, otherwise the switch-off parity leg and per-lever "
                f"attribution silently do not run"
            )
            # Only the manifest feeds switch_env, so an enablement round may instead
            # arm its gate through the proposal. One that is armed nowhere would
            # bench inert and reproduce the same failure, so it is still refused.
            is_enablement = bool(params.get("enablement"))
            unarmed_gates = [g for g in undeclared_gates if g not in proposal_extra_envs]
            if not is_enablement or unarmed_gates:
                if is_enablement:
                    reason = (
                        f"patch gates on environment switch(es) "
                        f"{', '.join(unarmed_gates)} that nothing turns on: declare "
                        f"them in the '{_switch_manifest.MANIFEST_KEY}' manifest or "
                        f"set them in the proposal, otherwise the patch benches inert"
                    )
                log.warning("integrate_patch: %s", reason)
                return {
                    "status": "reverted",
                    "error_class": "framework_switch_gates_undeclared",
                    "error": reason,
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "config_changes_applied": {},
                    "reason": reason,
                    "framework_switch_problems": switch_problems + [reason],
                    "undeclared_switch_gates": undeclared_gates,
                }
            log.info("integrate_patch(enablement): %s — armed by the proposal, benching", reason)
            switch_problems.append(reason)
        if switch_manifest and not patch_paths:
            # A manifest without a patch describes switches that gate code which
            # was never delivered. Setting them would be a no-op, and registering
            # them as levers would leave the ledger pointing at absent code.
            switch_problems.append(
                f"discarded {len(switch_manifest)} switch(es): the deliverable carries no patch, "
                f"so there is no rewrite for them to gate"
            )
            switch_manifest = []
        if switch_manifest:
            proposal_extra_envs.update(_switch_manifest.switch_env(switch_manifest))
            log.info(
                "integrate_patch: benching with %d framework rewrite switch(es) on\n%s",
                len(switch_manifest),
                _switch_manifest.summarize(switch_manifest, switch_problems),
            )
        elif switch_problems:
            log.warning(
                "integrate_patch: framework switch manifest unusable\n%s",
                _switch_manifest.summarize(switch_manifest, switch_problems),
            )
        ctx._ip_switch_manifest = switch_manifest  # type: ignore[attr-defined]
        ctx._ip_switch_problems = switch_problems  # type: ignore[attr-defined]

        explicit_artifacts = params.get("artifacts")
        artifact_specs, artifact_resolve_errors = _resolve_artifact_specs(
            specialist_workspace=specialist_workspace,
            explicit_artifacts=(list(explicit_artifacts) if isinstance(explicit_artifacts, list) else None),
            done_payload=done_payload,
        )
        target_gpu_type = str(
            getattr(shared_state, "gpu_type", "") or params.get("gpu_type") or params.get("target_platform") or ""
        ).strip()
        artifact_runtime_errors = _validate_aiter_gemm_artifacts(
            artifact_specs,
            gpu_type=target_gpu_type,
        )
        if artifact_runtime_errors:
            await self._maybe_write_framework_kb_record(
                params=params,
                done_payload=done_payload,
                outcome="rejected_apply_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            return {
                "status": "apply_failed",
                "error_class": "artifact_not_runtime_ready",
                "error": artifact_runtime_errors,
                "artifact_errors": artifact_runtime_errors,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_applied": [],
                "config_changes_applied": {},
                "reason": (
                    "AITER GEMM model-config artifacts must contain measured, "
                    "non-placeholder rows for the target GPU architecture and CU count"
                ),
            }

        _setup_ran = bool(setup_result.get("applied"))
        if (
            not patch_paths
            and not proposal_extra_args
            and not proposal_extra_envs
            and not artifact_specs
            and not _setup_ran
        ):
            # Launch-only mode: skip the no-patches early-return and fall through to bench.
            if params.get("enablement_launch_only"):
                self._replay_base_artifacts(params)
                output_root = runs_dir(self.session_dir, "integrate_patch", specialist_task_id)
                output_root.mkdir(parents=True, exist_ok=True)
                ctx._ip_framework_root = None  # type: ignore[attr-defined]
                ctx._ip_stash_state = "clean"  # type: ignore[attr-defined]
                ctx._ip_stash_note = ""  # type: ignore[attr-defined]
                ctx._ip_applied = []  # type: ignore[attr-defined]
                ctx._ip_applied_artifacts = []  # type: ignore[attr-defined]
                self._publish_gate_state(
                    ctx,
                    output_root=output_root,
                    config_changes_applied={},
                    extra_server_args_applied="",
                    extra_envs_applied={},
                    dropped_env_overrides=[],
                    setup_result=setup_result,
                )
                return None
            _no_patches: dict[str, Any] = {
                "status": "no_patches",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "artifacts_applied": [],
                "artifact_errors": artifact_resolve_errors,
                "setup_commands_applied": list(setup_result.get("applied") or []),
                "setup_commands_skipped": list(setup_result.get("skipped") or []),
                "reason": _with_skipped_setup_reason(
                    "neither patches, config_changes, installable artifacts, nor "
                    "allowlisted setup commands were supplied / discoverable for "
                    "this specialist task",
                    setup_result,
                ),
            }
            if params.get("enablement"):
                _no_patches["enablement"] = True
            # Forward grounding-drop details so framework.py can surface them in
            # the next round's mandate.  The field lives on done_payload (written
            # by runner.py) and must be forwarded here because _no_patches is the
            # concrete dict framework.py reads via _maybe_rearm_enablement.
            grounding_drops = (done_payload or {}).get("patches_dropped_by_grounding")
            if isinstance(grounding_drops, list) and grounding_drops:
                _no_patches["patches_dropped_by_grounding"] = grounding_drops
            return _no_patches

        explicit_framework_root = str(params.get("framework_source_root") or "").strip() or None
        framework_root = _resolve_framework_root(
            explicit_framework_root,
            patch_paths=patch_paths,
            recorded_root=_sole_patch_root(done_payload),
        )
        if patch_paths and framework_root is None:
            _lane_early = _derive_lane(params)
            if explicit_framework_root:
                # A non-allowlisted root is refused on that ground alone; only
                # an allowlisted one that simply lacks the files is the patches' fault.
                allowed_root = allowlisted_explicit_root(explicit_framework_root)
                if allowed_root is not None:
                    if not _read_patch_texts(patch_paths):
                        _error_class = "patch_unreadable"
                        _error = "no patch file could be read; verify paths and permissions"
                    else:
                        missing_records = _preflight_missing_targets(allowed_root, patch_paths)
                        if missing_records:
                            _error_class = "patch_target_missing"
                            _error = missing_records
                        else:
                            _error_class = "framework_source_root_rejected"
                            _error = (
                                f"framework_source_root {explicit_framework_root!r} could not "
                                "be unambiguously matched to the patch targets"
                            )
                else:
                    _error_class = "framework_source_root_rejected"
                    _error = (
                        f"framework_source_root {explicit_framework_root!r} is not "
                        "under the configured trusted source scope"
                    )
            else:
                _error_class = "no_framework_agent_root"
                _error = (
                    "no framework_source_root resolved; cannot apply "
                    "patches. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                )
            _early: dict[str, Any] = {
                "status": "apply_failed",
                "error_class": _error_class,
                "error": _error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "lane": _lane_early,
                "retry_feedback": [],
                "prior_patches": [str(p) for p in patch_paths],
            }
            if _error_class == "patch_target_missing":
                _early["advisory"] = (
                    "patch target file(s) absent from framework_source_root "
                    f"{explicit_framework_root}; author patches only against files "
                    "that exist in the installed framework tree."
                )
            # A set spanning two trees is exactly what fails root resolution here,
            # so this is the result that has to carry the reason onward.
            if (done_payload or {}).get("patches_span_multiple_roots"):
                _early["patches_span_multiple_roots"] = True
            if params.get("enablement"):
                _early["enablement"] = True
            return _early

        if patch_paths and framework_root is not None:
            missing_records = _preflight_missing_targets(framework_root, patch_paths)
            if missing_records:
                await self._maybe_write_framework_kb_record(
                    params=params,
                    done_payload=done_payload,
                    outcome="rejected_apply_fail",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                _lane_missing = _derive_lane(params)
                _missing_result: dict[str, Any] = {
                    "status": "apply_failed",
                    "error_class": "patch_target_missing",
                    "error": missing_records,
                    "advisory": (
                        "patch target file(s) absent from framework_source_root "
                        f"{framework_root}; author patches only against files that "
                        "exist in the installed framework tree (inspect it with "
                        "Glob/Grep before writing the diff)."
                    ),
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "config_changes_applied": {},
                    "lane": _lane_missing,
                    "retry_feedback": [],
                    "prior_patches": [str(p) for p in patch_paths],
                }
                if params.get("enablement"):
                    _missing_result["enablement"] = True
                return _missing_result

        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Write pending_integrate sentinel before any framework tree mutation.
        # The Coordinator clears this after promoting the final result.
        # shared_state.save() is called here (executor owns the sentinel write;
        # the Coordinator owns state promotion on completion).
        if shared_state is not None:
            try:
                shared_state.pending_integrate = {
                    "specialist_task_id": specialist_task_id,
                    "task_id": str(getattr(ctx.task, "task_id", "") or ""),
                    "patches": [str(p) for p in patch_paths],
                    "artifacts": [{"target": str(s.target), "rel_target": s.rel_target} for s in artifact_specs],
                    "config_changes": dict(config_changes),
                    "extra_server_args": proposal_extra_args,
                    "extra_envs": dict(proposal_extra_envs),
                    "framework_source_root": str(framework_root or ""),
                    "workspace": str(output_root),
                    # Attempt venv root for crash-resume GC.
                    "attempt_venv_root": str(getattr(ctx, "_ip_attempt_venv_root", "") or ""),
                    "ts": _now_iso(),
                }
                shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — sentinel is best-effort
                log.exception("integrate_patch: failed to persist pending_integrate sentinel")

        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            log.error(
                "integrate_patch: cannot stash user changes in %s: %s; aborting to avoid data loss",
                framework_root,
                stash_note,
            )
            return {
                "status": "apply_failed",
                "error_class": "stash_failed",
                "error": f"refusing to proceed: user changes could not be stashed ({stash_note})",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }

        # The stash is on the stack and the tree is about to be mutated, so
        # ``__call__``'s undo has to be able to see both before anything writes.
        ctx._ip_framework_root = framework_root  # type: ignore[attr-defined]
        ctx._ip_stash_state = stash_state  # type: ignore[attr-defined]
        ctx._ip_stash_note = stash_note  # type: ignore[attr-defined]

        self._replay_base_artifacts(params)

        git_tree = _is_git_tree(framework_root) if framework_root is not None else False
        self._nogit_patch_backups: list[dict[str, Any]] = []

        applied: list[Path] = []
        applied_artifacts: list[dict[str, Any]] = []
        apply_errors: list[dict[str, str]] = []
        apply_feedbacks: list[ApplyFeedback] = []
        # ``applied`` is published by identity and appended to in place.
        ctx._ip_applied = applied  # type: ignore[attr-defined]
        ctx._ip_applied_artifacts = applied_artifacts  # type: ignore[attr-defined]
        self._apply_attempted = bool(patch_paths)
        for patch in patch_paths:
            # ``vet_patches`` runs at authoring time inside the specialist
            # runner, so a patch from anywhere else -- ``params.patches``, and
            # every remotely-fetched ``upstream_pr`` diff -- arrives unvetted.
            try:
                patch_text = patch.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                apply_errors.append({"patch": str(patch), "stderr": f"unreadable: {exc!r}"})
                break
            if not is_unified_diff(patch_text):
                apply_errors.append({"patch": str(patch), "stderr": "not a unified diff"})
                break
            escaping = patch_escapes_tree(patch_text)
            if escaping is not None:
                apply_errors.append({"patch": str(patch), "stderr": f"path escapes tree: {escaping!r}"})
                break
            if git_tree:
                ok, err, fb = _git_apply_collect_feedback(framework_root, patch, three_way=False)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            else:
                nogit_backup_root = output_root / "patch_backups"
                ok, err, backups, fb = _apply_patch_no_git(
                    framework_root,
                    patch,
                    nogit_backup_root,
                    seq_offset=len(self._nogit_patch_backups),
                )
                self._nogit_patch_backups.extend(backups)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            applied.append(patch)

        if apply_errors:
            reverted = self._revert_patches(framework_root, applied)
            await self._maybe_write_framework_kb_record(
                params=params,
                done_payload=done_payload,
                outcome="rejected_apply_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            lane = _derive_lane(params)
            base_result: dict[str, Any] = {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "workspace": str(output_root),
                "lane": lane,
                "retry_feedback": [fb.to_dict() for fb in apply_feedbacks],
                "prior_patches": [str(p) for p in patch_paths],
            }
            if bool(params.get("enablement")):
                base_result["enablement"] = True
            return _with_stash_restore(framework_root, stash_state, stash_note, base_result)

        if artifact_specs:
            applied_artifacts, artifact_apply_errors = self._apply_artifacts(
                artifact_specs,
                backup_root=output_root / "artifact_backups",
            )
            ctx._ip_applied_artifacts = applied_artifacts  # type: ignore[attr-defined]
            if artifact_apply_errors:
                self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                await self._maybe_write_framework_kb_record(
                    params=params,
                    done_payload=done_payload,
                    outcome="rejected_apply_fail",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "apply_failed",
                        "error_class": "artifact_install_failed",
                        "error": artifact_resolve_errors + artifact_apply_errors,
                        "specialist_task_id": specialist_task_id,
                        "patches_applied": [],
                        "patches_reverted": [str(p) for p in reverted],
                        "artifacts_applied": [],
                        "config_changes_applied": {},
                        "workspace": str(output_root),
                    },
                )

        extra_server_args_applied = proposal_extra_args
        extra_envs_applied = dict(proposal_extra_envs)
        config_changes_applied = dict(extra_envs_applied)

        if params.get("apply_only"):
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "applied_no_bench",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [str(p) for p in applied],
                    "patches_reverted": [],
                    "artifacts_applied": applied_artifacts,
                    "config_changes_applied": config_changes_applied,
                    "extra_server_args_applied": extra_server_args_applied,
                    "extra_envs_applied": extra_envs_applied,
                    "dropped_env_overrides": dropped_env_overrides,
                    "reason": "apply_only=True; benchmark skipped",
                    "workspace": str(output_root),
                },
            )

        # The tree-mutation values are already published above, as the tree took
        # them; what is left is what only the gate reads.
        self._publish_gate_state(
            ctx,
            output_root=output_root,
            config_changes_applied=config_changes_applied,
            extra_server_args_applied=extra_server_args_applied,
            extra_envs_applied=extra_envs_applied,
            dropped_env_overrides=dropped_env_overrides,
            setup_result=setup_result,
        )
        return None

    async def _stage_gate(
        self,
        ctx: Any,
        params: dict[str, Any],
        extra: dict[str, Any],
        *,
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        dropped_env_overrides: list[str],
        setup_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Bench + enablement/perf KEEP/REVERT gate.

        Runs _bench_patch, applies the appropriate gate, and returns the
        final integration result. Never returns None.
        """
        # Activate the provisioned attempt runtime by threading its YAML-layer
        # override into params so both bench wirings pick it up.
        provision_result = getattr(ctx, "_ip_provision_result", None)
        if provision_result is not None and getattr(provision_result, "ok", False):
            params = dict(params)
            params["runtime_override"] = provision_result.runtime.to_runtime_override()

        # Bound both bench legs by the session wall-clock, as the sweep and explore
        # arms already are: the declared cap answers "how long before this counts
        # as hung", not "how much budget is left", so without this a patch benched
        # near the end of a run could outlive the run itself. Resolved once here
        # and reused by the parity leg -- the deadline is an absolute monotonic
        # timestamp, so it stays correct as the gate progresses.
        session_deadline_sec, variant_expected_sec = session_grid_bounds(shared_state)
        try:
            bench_result, gate_evidence = await self._bench_patch(
                params=params,
                output_root=output_root,
                extra_server_args_applied=extra_server_args_applied,
                extra_envs_applied=extra_envs_applied,
                specialist_task_id=specialist_task_id,
                state_model_path=str(getattr(shared_state, "model_path", "") or ""),
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
        except FrameworkScriptMismatchError as exc:
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "error_class": "framework_script_mismatch",
                    "error": str(exc),
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "reason": str(exc),
                    "workspace": str(output_root),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "error_class": "bench_exception",
                    "error": repr(exc),
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "config_changes_applied": {},
                    "reason": f"bench raised: {exc!r}",
                    "workspace": str(output_root),
                },
            )

        if params.get("enablement"):
            verdict = await self._gate_enablement(
                params=params,
                extra=extra,
                specialist_task_id=specialist_task_id,
                done_payload=done_payload,
                output_root=output_root,
                framework_root=framework_root,
                stash_state=stash_state,
                stash_note=stash_note,
                applied=applied,
                applied_artifacts=applied_artifacts,
                config_changes_applied=config_changes_applied,
                extra_server_args_applied=extra_server_args_applied,
                extra_envs_applied=extra_envs_applied,
                setup_result=setup_result,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
                ctx=ctx,
            )
        else:
            verdict = await self._gate_perf(
                params=params,
                extra=extra,
                specialist_task_id=specialist_task_id,
                shared_state=shared_state,
                done_payload=done_payload,
                output_root=output_root,
                framework_root=framework_root,
                stash_state=stash_state,
                stash_note=stash_note,
                applied=applied,
                applied_artifacts=applied_artifacts,
                config_changes_applied=config_changes_applied,
                extra_server_args_applied=extra_server_args_applied,
                extra_envs_applied=extra_envs_applied,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
                ctx=ctx,
            )
        if dropped_env_overrides:
            verdict["dropped_env_overrides"] = dropped_env_overrides
        return verdict

    async def _gate_enablement(
        self,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        setup_result: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
        ctx: Any = None,
    ) -> dict[str, Any]:
        """Enablement gate: runnability + minimal-correctness.

        The verdict depends on the trigger origin, because the two origins have
        different evidence available:

        * ``accuracy >= floor`` -> ``correctness_ok=True`` (KEEP, verified).
          eval-origin additionally requires the score to carry a task + metric;
          without them it did not come from a real eval and fails closed.
        * present but below floor / non-positive / non-finite ->
          ``correctness_ok=False`` (REVERT, garbage output).
        * ``accuracy is None`` -> eval-origin fails closed
          (``correctness_ok=False``): the trigger *was* an accuracy failure, so a
          candidate that produces no score has not shown it fixed anything.
          boot-origin stays ``None`` (KEEP but provisional) — it only ever
          claimed to make the model boot, and eval-less runs must not be blocked.

        On KEEP the benched env/arg layers are reported as
        ``enablement_effective_config``, captured from the variant that ran: the
        materialized YAML holds only the base layer, so the revalidation baseline
        needs them to re-run the graded config.
        When an attempt runtime was provisioned, the stack action is recorded in
        the result (``enablement_kept_stack_action``) so it survives rearm. On
        REVERT / non-KEEP, the attempt runtime dir is GC'd.

        Every verdict carries ``framework_switch_problems`` (auditable switch-gate
        demotion record) and ``framework_root`` (the source tree patches were applied
        against, needed to replay them on a fresh machine).
        """
        stack_action = getattr(ctx, "_ip_stack_action", None) if ctx is not None else None
        provision_result = getattr(ctx, "_ip_provision_result", None) if ctx is not None else None
        switch_problems: list[str] = list(getattr(ctx, "_ip_switch_problems", None) or [])

        def _gc_on_revert() -> None:
            """GC the attempt runtime dir on a non-KEEP enablement outcome."""
            root = str(getattr(ctx, "_ip_attempt_venv_root", "") or "") if ctx is not None else ""
            if root:
                # venv_root is ``<attempt_dir>/venv``; GC the whole attempt dir.
                self._gc_attempt_dir(Path(root).parent)

        from hyperloom.agents.framework.enablement import (
            FailureSignature,
            classify_failure,
            enablement_made_progress,
            runnable_decision,
        )

        new_tput = bench_result.get("output_throughput")
        booted = isinstance(new_tput, (int, float)) and new_tput > 0
        probe_timed_out = bool(gate_evidence.get("timed_out"))

        enablement_accuracy = gate_evidence.get("enablement_accuracy")
        _param_floor = params.get("enablement_accuracy_floor")
        floor = float(_param_floor) if isinstance(_param_floor, (int, float)) else DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        eval_origin = _is_eval_origin(params)
        accuracy_kind = classify_accuracy_failure(enablement_accuracy, floor)
        correctness_ok: bool | None
        if enablement_accuracy is None:
            # Truly absent: eval-origin fails closed; boot-origin stays provisional.
            correctness_ok = False if eval_origin else None
        elif accuracy_meets_floor(enablement_accuracy, floor):
            correctness_ok = True
        else:
            # Present but below floor / non-positive / non-finite.
            correctness_ok = False
        # eval-origin only: a score with no task/metric did not come from a real
        # eval, so it cannot clear the gate. This reads the candidate's OWN run
        # (both keys are stamped beside the accuracy it is judging), unlike the
        # contract fingerprint it replaces: RUN_EVAL is itself a hashed contract
        # field, so an eval-less re-baseline could poison the stored digest and
        # veto every later candidate without ever consulting its accuracy.
        if (
            eval_origin
            and correctness_ok
            and not (gate_evidence.get("enablement_accuracy_task") and gate_evidence.get("enablement_accuracy_metric"))
        ):
            correctness_ok = False
            log.warning(
                "integrate_patch: eval-origin accuracy %s carries no task/metric; reverting",
                enablement_accuracy,
            )
        eval_provenance = {
            "enablement_origin": str(params.get("enablement_origin") or ""),
            "enablement_observed_accuracy": enablement_accuracy,
            "enablement_accuracy_floor": floor,
            "accuracy_task": gate_evidence.get("enablement_accuracy_task") or "",
            "accuracy_metric": gate_evidence.get("enablement_accuracy_metric") or "",
            "enablement_eval_failure_kind": accuracy_kind or "",
        }

        after_signature = classify_failure(str(bench_result.get("error") or ""))
        before_signature: FailureSignature | None = None
        raw_before = params.get("enablement_before_signature")
        if isinstance(raw_before, dict):
            try:
                before_signature = FailureSignature(**raw_before)
            except (TypeError, ValueError):
                before_signature = None

        runs, run_reason = runnable_decision(
            probe_returncode=0 if booted else 1,
            correctness_ok=correctness_ok,
            probe_timed_out=probe_timed_out,
            before_signature=before_signature,
            after_signature=after_signature,
        )
        if not runs:
            advanced = (not booted) and enablement_made_progress(before_signature, after_signature)
            if advanced:
                stacked_patches = [str(p) for p in applied]
                new_log = str(bench_result.get("error") or "")
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                await self._maybe_write_framework_kb_record(
                    params=params,
                    done_payload=done_payload,
                    outcome="integrated",
                    tps_delta_pct=0.0,
                    extra=extra,
                )
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "advanced",
                        "specialist_task_id": specialist_task_id,
                        "patches_applied": stacked_patches,
                        "patches_reverted": [str(p) for p in reverted],
                        "artifacts_applied": applied_artifacts,
                        "artifacts_reverted": artifacts_reverted,
                        "config_changes_applied": config_changes_applied,
                        "extra_envs_applied": extra_envs_applied,
                        "extra_server_args_applied": extra_server_args_applied,
                        "framework_switch_problems": switch_problems,
                        "framework_root": str(framework_root or ""),
                        "output_throughput": new_tput,
                        "enablement": True,
                        "advanced": True,
                        "runnable": False,
                        "correctness_verified": False,
                        "reason": _with_skipped_setup_reason(
                            f"enablement progressed: {run_reason}; boot advanced "
                            f"to a new gap ({after_signature.kind}) — patch recorded "
                            f"as a base for the next round",
                            setup_result,
                        ),
                        "after_signature": after_signature.to_dict(),
                        "enablement_launch_log": new_log,
                        "setup_commands_applied": list(setup_result.get("applied") or []),
                        "setup_commands_skipped": list(setup_result.get("skipped") or []),
                        "bench_result": bench_result,
                        "workspace": str(output_root),
                        **eval_provenance,
                    },
                )
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            _gc_on_revert()
            await self._maybe_write_framework_kb_record(
                params=params,
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "framework_switch_problems": switch_problems,
                    "framework_root": str(framework_root or ""),
                    "output_throughput": new_tput,
                    "enablement": True,
                    "runnable": False,
                    "correctness_verified": correctness_ok is True,
                    # The round ran and the boot still did not come up. When the
                    # specialist's own setup commands were dropped on the way in,
                    # that is the likeliest reason -- and the one the next round
                    # needs, since re-authoring the same proposal cannot help.
                    "reason": _with_skipped_setup_reason(f"enablement not runnable: {run_reason}", setup_result),
                    "setup_commands_applied": list(setup_result.get("applied") or []),
                    "setup_commands_skipped": list(setup_result.get("skipped") or []),
                    "bench_result": bench_result,
                    "workspace": str(output_root),
                    **eval_provenance,
                },
            )

        provisional = correctness_ok is None
        reason = f"enablement runnable: {run_reason}"
        if provisional:
            reason += " (provisional: booted but eval produced no accuracy; correctness not verified)"
        await self._maybe_write_framework_kb_record(
            params=params,
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=0.0,
            extra=extra,
        )
        kept_result: dict[str, Any] = {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "artifacts_applied": applied_artifacts,
            "config_changes_applied": config_changes_applied,
            "extra_server_args_applied": extra_server_args_applied,
            "extra_envs_applied": extra_envs_applied,
            "framework_switch_problems": switch_problems,
            "framework_root": str(framework_root or ""),
            "output_throughput": new_tput,
            "enablement": True,
            "runnable": True,
            "correctness_verified": correctness_ok is True,
            "provisional": provisional,
            "reason": _with_skipped_setup_reason(reason, setup_result),
            "setup_commands_applied": list(setup_result.get("applied") or []),
            "setup_commands_skipped": list(setup_result.get("skipped") or []),
            "bench_result": bench_result,
            "workspace": str(output_root),
            # Base YAML only; the env/arg layers live in enablement_effective_config.
            "enablement_accepted_config_path": str(bench_result.get("materialized_config") or ""),
            # Captured from the variant this leg launched, so a revalidation
            # replays the graded configuration rather than a re-derived one.
            "enablement_effective_config": dict(bench_result.get("effective_config") or {}),
            **eval_provenance,
        }
        # Record the KEEP'd attempt runtime so it survives rearm and every later
        # bench in this session re-activates it.
        if stack_action is not None and provision_result is not None and getattr(provision_result, "ok", False):
            kept_result["enablement_kept_stack_action"] = stack_action.to_state()
            kept_result["enablement_active_runtime"] = provision_result.runtime.to_state()
            kept_result["installed_versions"] = dict(getattr(provision_result, "installed_versions", {}) or {})
        # Editable-refresh the localized closure + snapshot a manifest that
        # survives rearm so the closure is recorded and not re-fetched.
        manifest = self._finalize_localization_keep(
            ctx,
            framework_root=framework_root,
            specialist_task_id=specialist_task_id,
            provision_result=provision_result,
        )
        if manifest:
            kept_result["enablement_localization_manifest"] = manifest
        return _with_stash_restore(framework_root, stash_state, stash_note, kept_result)

    def _finalize_localization_keep(
        self,
        ctx: Any,
        *,
        framework_root: Path | None,
        specialist_task_id: str,
        provision_result: Any,
    ) -> dict[str, Any]:
        """Editable-refresh a localized closure and snapshot its manifest.

        Runs the framework adapter's editable-refresh argv against the attempt
        interpreter (best-effort; skipped when there is no attempt runtime or no
        refresh argv), then records a localization manifest via
        :func:`snapshot_source_layer`. Returns the manifest dict (empty when no
        localization ran).
        """
        touched = list(getattr(ctx, "_ip_localization_touched", None) or [])
        if not touched or framework_root is None:
            return {}
        action = getattr(ctx, "_ip_stack_action", None)
        # Editable-refresh so localized Python changes take effect in the attempt
        # runtime (no-op for plain wheel trees like atom).
        try:
            from ...framework.adapters import get_adapter

            venv_py = ""
            if provision_result is not None and getattr(provision_result, "ok", False):
                venv_py = str(getattr(provision_result.runtime, "python_path", "") or "")
            fw = str(getattr(action, "framework", "") or "")
            argv = get_adapter(fw).editable_refresh_argv(venv_py, str(framework_root)) if venv_py else None
            if argv:
                subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)  # noqa: S603
        except Exception:  # noqa: BLE001 — refresh is best-effort
            log.debug("integrate_patch: localization editable-refresh failed", exc_info=True)
        # Manifest via the existing snapshot mechanism.
        try:
            from ...source_snapshot import snapshot_source_layer

            base_sha = ""
            _cp = _run_git_cp(["-C", str(framework_root), "rev-parse", "HEAD"], timeout=30.0)
            if _cp is not None and getattr(_cp, "returncode", 1) == 0:
                base_sha = (_cp.stdout or "").strip()
            dest = self.session_dir / "optimization_stack" / "localization" / (specialist_task_id or "keep")
            snap = snapshot_source_layer(
                framework_root=framework_root,
                base_sha=base_sha,
                rel_paths=touched,
                dest_dir=dest,
                provenance="localization",
                extra={
                    "specialist_task_id": specialist_task_id,
                    "kind": str(getattr(action, "kind", "") or ""),
                    "repo_url": str(getattr(action, "repo_url", "") or ""),
                    "pr_number": int(getattr(action, "pr_number", 0) or 0),
                },
            )
            return dict(snap) if snap else {}
        except Exception:  # noqa: BLE001 — manifest is best-effort durability
            log.exception("integrate_patch: localization snapshot failed")
            return {}

    async def _gate_perf(
        self,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        shared_state: Any,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        config_changes_applied: dict[str, str],
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        """Throughput KEEP / REVERT decision, or no verdict when the run stopped it."""
        # Grade against the current live anchor, not a stale task snapshot.
        base_tput, anchor_drifted = resolve_anchor_with_drift(
            float(params.get("base_tput") or 0.0),
            shared_state,
        )
        if anchor_drifted:
            log.warning(
                "integrate_patch: anchor drift; grading against live anchor %.1f",
                base_tput,
            )

        keep_threshold_pct = float(params.get("keep_threshold_pct", self.keep_threshold_pct))

        stopped = stopped_by_the_run_class(bench_result.get("error_class"))
        if stopped is not None:
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "failed",
                    "error_class": stopped.error_class,
                    "error": stopped.interrupted,
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "bench_result": bench_result,
                    "workspace": str(output_root),
                },
            )

        # ``new_tput`` is reported as ``output_throughput``; the KEEP gate is
        # graded on whichever axis this session uses, both sides from one
        # resolver. On the output axis the drift-resolved ``base_tput`` is the
        # reference, which is what resolve_anchor_with_drift exists for.
        new_tput = bench_result.get("output_throughput")
        graded = resolve_graded_comparison(shared_state, bench_result)
        if graded.degrade_reason:
            log.info("integrate_patch: grading on output throughput (%s)", graded.degrade_reason)
        if not graded.graded_on_total:
            delta_pct = gain_pct(new_tput, base_tput)
        elif graded.vetoed:
            log.info("integrate_patch: candidate failed the interactivity constraint")
            delta_pct = None
        else:
            delta_pct = gain_pct(graded.candidate, graded.reference)

        accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
        fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
        acc_required = bool(params.get("require_accuracy_for_keep", fw_authored))
        acc_baseline = params.get("accuracy_baseline")
        if acc_required and not acc_baseline:
            _ss = extra.get("shared_state") or extra.get("state")
            if _ss is not None:
                acc_baseline = getattr(_ss, "baseline_accuracy", None)
        acc_block, acc_reason, acc_degraded = accuracy_keep_block(
            accuracy_pass,
            required=acc_required,
            baseline_accuracy=acc_baseline,
        )
        if acc_degraded:
            log.warning(
                "integrate_patch: accuracy gate required but no baseline accuracy; "
                "KEEP allowed on throughput only (task=%s)",
                specialist_task_id,
            )
        gate_pass = delta_pct is not None and delta_pct >= keep_threshold_pct and not acc_block
        _ss_kb = extra.get("shared_state") or extra.get("state")
        acc_delta_pct = _accuracy_delta_pct(
            gate_evidence.get("accuracy"),
            acc_baseline or getattr(_ss_kb, "baseline_accuracy", None),
        )
        cfg_fingerprint = canonical_fingerprint(
            params.get("extra_server_args"),
            params.get("extra_envs"),
        )

        switch_manifest: list[dict[str, Any]] = list(getattr(ctx, "_ip_switch_manifest", None) or [])
        switch_problems: list[str] = list(getattr(ctx, "_ip_switch_problems", None) or [])

        # Switch-off parity. Run before either KEEP verdict, since both of them
        # leave the patch on disk and therefore both depend on it being inert when
        # disabled.
        #
        # It runs on a quality regression too, which looks wasteful and is not: the
        # switches are benched together, so a moved output localises to the bundle
        # rather than to a switch. On a live session a four-switch bundle reached
        # +65.5% and was reverted whole on the gate, discarding three switches that
        # were never implicated along with the one that was. Default-off code costs
        # nothing to keep and explore can bisect it per lever — but only if the tree
        # is genuinely unchanged with every switch unset, which is exactly what this
        # leg measures. An unswitched patch has no "off" state to fall back to, so
        # it still reverts without spending the leg.
        parity: dict[str, Any] = {"ran": False, "ok": True, "reason": ""}
        if switch_manifest:
            # The parity leg is an additional full bench, so it needs the same
            # session bound the first bench got. Resolved here rather than
            # threaded from the caller because the deadline is an absolute
            # monotonic timestamp: the budget the first bench spent is already
            # reflected in it. Only this leg reads it, so it is resolved only
            # when the leg runs.
            session_deadline_sec, variant_expected_sec = session_grid_bounds(shared_state)
            parity = await self._switch_off_parity(
                params=params,
                output_root=output_root,
                specialist_task_id=specialist_task_id,
                switch_manifest=switch_manifest,
                base_tput=base_tput,
                state_model_path=str(getattr(shared_state, "model_path", "") or ""),
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
            if not parity.get("ok"):
                # An unmeasurable parity leg reverts under its own verdict: the patch
                # was never shown to break, so neither the log line nor the KB lesson
                # may say that it did.
                from ...knowledge import kb_writeback as _kb

                inconclusive = bool(parity.get("inconclusive"))
                error_class = "switch_off_parity_inconclusive" if inconclusive else "switch_off_parity_failed"
                kb_outcome = (
                    _kb.OUTCOME_REVERTED_PARITY_INCONCLUSIVE if inconclusive else _kb.OUTCOME_REVERTED_SWITCH_OFF_PARITY
                )
                artifacts_reverted = self._revert_artifacts(applied_artifacts)
                reverted = self._revert_patches(framework_root, applied)
                log.warning(
                    "integrate_patch: switch-off parity %s task=%s: %s",
                    "INCONCLUSIVE" if inconclusive else "FAILED",
                    specialist_task_id,
                    parity.get("reason"),
                )
                await self._maybe_write_framework_kb_record(
                    params=params,
                    done_payload=done_payload,
                    outcome=kb_outcome,
                    tps_delta_pct=float(delta_pct or 0.0),
                    extra=extra,
                    accuracy_delta_pct=acc_delta_pct,
                    config_fingerprint=cfg_fingerprint,
                )
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "reverted",
                        "error_class": error_class,
                        "specialist_task_id": specialist_task_id,
                        "patches_applied": [],
                        "patches_reverted": [str(p) for p in reverted],
                        "artifacts_reverted": artifacts_reverted,
                        "config_changes_applied": {},
                        "output_throughput": new_tput,
                        "delta_pct": delta_pct,
                        "accuracy_pass": accuracy_pass,
                        "base_tput": base_tput,
                        "keep_threshold_pct": keep_threshold_pct,
                        "reason": str(parity.get("reason") or "switch-off parity failed"),
                        "switch_off_parity": parity,
                        "framework_switch_problems": switch_problems,
                        "bench_result": bench_result,
                        "workspace": str(output_root),
                    },
                )

        if not gate_pass:
            # Two-tier verdict for a framework-rewrite patch. Every rewrite in it
            # is behind a switch that defaults OFF, so keeping the code with the
            # switches unset changes nothing at runtime — which makes reverting
            # it the more expensive choice. The bundle failed as a bundle, but a
            # bundle usually mixes rewrites that pay with one that does not, and
            # some of them are enablers that cannot pay until measured together
            # with what they unlock. So keep the code inert, register the switches
            # as levers, and let the explore phase find the subset that wins.
            #
            # A quality regression does not condemn the bundle either, for the same
            # reason: the switches are benched together, so a moved output says
            # "at least one of these is wrong", not "all of them are". The bundle
            # stays inert and explore bisects it per lever — the verdict carries
            # ``quality_unverified`` so nothing downstream mistakes it for a clean
            # keep. What still condemns it is failing parity, handled above: code
            # that is not inert when disabled would skew every later measurement,
            # and that check now runs on this path too.
            if switch_manifest and applied:
                return await self._keep_inert_switches(
                    params=params,
                    extra=extra,
                    specialist_task_id=specialist_task_id,
                    done_payload=done_payload,
                    output_root=output_root,
                    framework_root=framework_root,
                    stash_state=stash_state,
                    stash_note=stash_note,
                    applied=applied,
                    applied_artifacts=applied_artifacts,
                    switch_manifest=switch_manifest,
                    switch_problems=switch_problems,
                    parity=parity,
                    bench_result=bench_result,
                    new_tput=new_tput,
                    delta_pct=delta_pct,
                    base_tput=base_tput,
                    keep_threshold_pct=keep_threshold_pct,
                    accuracy_pass=accuracy_pass,
                    acc_delta_pct=acc_delta_pct,
                    cfg_fingerprint=cfg_fingerprint,
                )
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(f"throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%")
            if acc_block and acc_reason:
                reasons.append(acc_reason)
            _probe_reason = eval_probe_summary(gate_evidence.get("eval_probe"))
            if _probe_reason:
                reasons.append(_probe_reason)
            _tput_ok = delta_pct is not None and delta_pct >= keep_threshold_pct
            revert_status = (
                "accuracy_unavailable_reject" if (acc_block and accuracy_pass is None and _tput_ok) else "reverted"
            )
            await self._maybe_write_framework_kb_record(
                params=params,
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=float(delta_pct or 0.0),
                extra=extra,
                accuracy_delta_pct=acc_delta_pct,
                config_fingerprint=cfg_fingerprint,
            )
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": revert_status,
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "output_throughput": new_tput,
                    "delta_pct": delta_pct,
                    "accuracy_pass": accuracy_pass,
                    "base_tput": base_tput,
                    "keep_threshold_pct": keep_threshold_pct,
                    "reason": "; ".join(reasons) or "gate failed",
                    "bench_result": bench_result,
                    "workspace": str(output_root),
                },
            )

        await self._maybe_write_framework_kb_record(
            params=params,
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=float(delta_pct or 0.0),
            extra=extra,
            accuracy_delta_pct=acc_delta_pct,
            config_fingerprint=cfg_fingerprint,
        )
        # The commit is what makes a KEEP survive the next candidate's
        # ``checkout --force HEAD -- . && clean -fd``; nothing replays the
        # source snapshot in-session. A failure is therefore terminal, or the
        # stack would claim a win the tree no longer carries. Non-git roots have
        # no checkout revert to survive, so the commit is skipped there.
        commit_failure: str = ""
        keep_committed = False
        if framework_root is None or not _is_git_tree(framework_root):
            log.info(
                "integrate_patch: non-git framework root %s; skipping commit-on-KEEP",
                framework_root,
            )
        else:
            try:
                touched = _patch_touched_paths(framework_root, applied)
                ok, note = _git_commit_kept(
                    framework_root,
                    f"hyperloom KEEP {specialist_task_id} ({delta_pct:+.2f}%)",
                    touched,
                )
                if not ok:
                    commit_failure = note or "git commit failed"
                else:
                    # Only a real commit advances HEAD, and ``_git_commit_kept``
                    # signals that with an empty note. Either no-op ("nothing to
                    # commit" or "no patch-touched paths to commit") leaves HEAD on
                    # the previous KEEP, so a later ``HEAD^..HEAD`` would be that
                    # KEEP's diff, not this one -- the harvest must not run then.
                    keep_committed = note == ""
            except Exception as exc:  # noqa: BLE001 — surfaced as a verdict below
                log.exception("integrate_patch: commit-on-KEEP raised")
                commit_failure = f"commit raised: {exc!r}"
        if commit_failure:
            log.error(
                "integrate_patch: commit-on-KEEP failed (%s); reverting rather than "
                "reporting a KEEP the next revert would silently remove",
                commit_failure,
            )
            artifacts_reverted = self._revert_artifacts(applied_artifacts)
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    # Applied, benched, then rolled back: the same terminal
                    # shape every other post-apply rollback reports. Calling it
                    # an apply failure sent the re-author loop after a diff
                    # that had already passed the bench.
                    "status": "reverted",
                    "error_class": "keep_commit_failed",
                    "error": commit_failure,
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "artifacts_reverted": artifacts_reverted,
                    "config_changes_applied": {},
                    "output_throughput": new_tput,
                    "delta_pct": delta_pct,
                    "bench_result": bench_result,
                    "reason": f"KEEP could not be committed: {commit_failure}",
                    "workspace": str(output_root),
                },
            )

        source_snapshot_dir = ""
        source_manifest_path = ""
        source_target_files: list[str] = []
        source_base_sha = ""
        source_snapshot_complete = False
        source_import_root_val = ""
        source_realized_patch = ""
        source_artifacts_outside_root = 0
        try:
            from ...source_snapshot import MANIFEST_NAME, snapshot_source_layer
            from ._patch_snapshot import _patch_touched_paths_split, harvest_realized_diff

            if framework_root is not None:
                _cp = _run_git_cp(["-C", str(framework_root), "rev-parse", "HEAD"], timeout=30.0)
                if _cp is not None and getattr(_cp, "returncode", 1) == 0:
                    source_base_sha = (_cp.stdout or "").strip()
                upserted_patch, deleted_patch = _patch_touched_paths_split(framework_root, applied)
                declared_ops = {r: "upsert" for r in upserted_patch}
                declared_ops.update({r: "delete" for r in deleted_patch})
                rel_paths = upserted_patch + deleted_patch
                # An artifact installed into a sibling tree is not addressable by
                # a rel path under this root, so it belongs to no snapshot here.
                # The count travels so a KEEP whose gain lives outside the tree
                # reads as a known gap rather than as a clean capture.
                inside_root = [
                    str(a["rel_target"])
                    for a in (applied_artifacts or [])
                    if isinstance(a, dict)
                    and a.get("rel_target")
                    and Path(str(a.get("root") or framework_root)).resolve() == framework_root.resolve()
                ]
                source_artifacts_outside_root = len(
                    [
                        a
                        for a in (applied_artifacts or [])
                        if isinstance(a, dict)
                        and a.get("rel_target")
                        and Path(str(a.get("root") or framework_root)).resolve() != framework_root.resolve()
                    ]
                )
                rel_paths += inside_root
                from ...framework.adapters import get_adapter

                source_import_root_val = get_adapter(str(params.get("framework") or "")).source_import_root(
                    str(framework_root)
                )
                dest = (
                    self.session_dir
                    / "optimization_stack"
                    / "src"
                    / (specialist_task_id or str(getattr(ctx.task, "task_id", "") or "keep"))
                )
                snap = snapshot_source_layer(
                    framework_root=framework_root,
                    base_sha=source_base_sha,
                    rel_paths=rel_paths,
                    dest_dir=dest,
                    provenance="integrate_patch",
                    extra={"specialist_task_id": specialist_task_id},
                    declared_ops=declared_ops,
                    import_root=source_import_root_val,
                )
                if snap:
                    source_snapshot_dir = str(snap.get("snapshot_dir") or "")
                    if source_snapshot_dir:
                        source_manifest_path = str(Path(source_snapshot_dir) / MANIFEST_NAME)
                    source_target_files = [
                        str(item.get("rel") or "")
                        for item in (snap.get("files") or [])
                        if isinstance(item, dict) and item.get("rel")
                    ]
                    source_snapshot_complete = bool(snap.get("complete"))
                    # Only harvest when a real commit landed. Without a new
                    # commit ``HEAD^..HEAD`` is the previous KEEP; harvesting it
                    # would publish that diff as this KEEP's realized change.
                    # Leaving it empty falls back to the delivered patch
                    # (realized=False), which is the honest record.
                    if keep_committed:
                        source_realized_patch = harvest_realized_diff(
                            framework_root,
                            rel_paths,
                            Path(source_snapshot_dir) / "realized.patch",
                        )
        except Exception:  # noqa: BLE001 — snapshot is best-effort durability
            log.exception("integrate_patch: source-layer snapshot failed")

        return _with_stash_restore(
            framework_root,
            stash_state,
            stash_note,
            {
                "status": "kept",
                "specialist_task_id": specialist_task_id,
                # Proposal ownership must survive delegated-result persistence
                # so resume replay cannot replace it with the then-current phase.
                "source_phase": str(params.get("source_phase") or ""),
                "domain": str(params.get("domain") or params.get("source_domain") or ""),
                "provenance": str(params.get("provenance") or ""),
                "gap_canonical_id": str(params.get("gap_canonical_id") or ""),
                "gap_layer": str(params.get("gap_layer") or ""),
                "framework_agent_authoring": bool(params.get("framework_agent_authoring")),
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "artifacts_applied": applied_artifacts,
                "config_changes_applied": config_changes_applied,
                "extra_server_args_applied": extra_server_args_applied,
                "extra_envs_applied": extra_envs_applied,
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": (f"throughput delta {delta_pct:+.2f}% >= {keep_threshold_pct:.2f}%"),
                "bench_result": bench_result,
                "workspace": str(output_root),
                "source_snapshot": source_snapshot_dir,
                "source_manifest": source_manifest_path,
                "source_snapshot_complete": source_snapshot_complete,
                "source_import_root": source_import_root_val,
                "source_realized_patch": source_realized_patch,
                "source_artifacts_outside_root": source_artifacts_outside_root,
                "target_files": source_target_files,
                "framework_root": str(framework_root or ""),
                "base_sha": source_base_sha,
                # The bundle cleared the gate, so its switches join the running
                # configuration and are registered as levers that are already on.
                # Attribution from here is leave-one-out.
                "framework_levers": switch_manifest,
                "framework_lever_outcome": ("default_on" if switch_manifest else ""),
                "framework_switch_problems": switch_problems,
                "switch_off_parity": parity,
            },
        )

    async def _switch_off_parity(
        self,
        *,
        params: dict[str, Any],
        output_root: Path,
        specialist_task_id: str,
        switch_manifest: list[dict[str, Any]],
        base_tput: float,
        state_model_path: str = "",
        session_deadline_sec: float | None = None,
        variant_expected_sec: float | None = None,
    ) -> dict[str, Any]:
        """Verify the patch is genuinely inert with every rewrite switch unset.

        The whole lever mechanism rests on one invariant: with no switch set, the
        patched tree behaves exactly like the original. That invariant is what
        makes it safe to keep unprofitable rewrite code on disk, what makes a
        per-lever measurement mean anything, and what keeps the baseline
        comparable across a session that has accumulated several rewrite patches.
        It is also the invariant an LLM is most likely to break by accident — by
        reading the switch once at import, inverting a default, or restructuring
        code outside the guard — and nothing else in the pipeline would notice: a
        switches-on bench that improves throughput looks like a success whether or
        not the switches-off path still works.

        So it is measured, not assumed. One extra leg with the switches removed
        must land inside a noise band around the pre-patch base.

        Args:
            params: The task params.
            output_root: The per-task workspace.
            specialist_task_id: The originating specialist.
            switch_manifest: Parsed switch manifest.
            base_tput: Pre-patch throughput to compare against.
            session_deadline_sec: Monotonic-clock session budget deadline for the
                parity bench, or ``None`` when unbounded.
            variant_expected_sec: Expected bench runtime, used to decide whether
                the remaining budget can fit the parity leg at all.

        Returns:
            ``{"ran", "ok", "tput", "delta_pct", "band_pct", "accuracy_pass",
            "reason"}``. ``ran`` is False when the check was skipped (disabled, or
            no usable base to compare against), which is reported rather than
            silently treated as a pass.
        """
        band_pct = float(params.get("switch_off_parity_band_pct", DEFAULT_SWITCH_OFF_PARITY_BAND_PCT))
        if not bool(params.get("enable_switch_off_parity", True)):
            return {"ran": False, "ok": True, "reason": "switch-off parity check disabled"}
        if base_tput <= 0:
            return {
                "ran": False,
                "ok": True,
                "reason": "no positive base throughput to compare a parity leg against",
            }
        switch_names = [entry["switch"] for entry in switch_manifest]
        try:
            parity_bench, parity_evidence = await self._bench_patch(
                params=params,
                output_root=output_root,
                extra_server_args_applied="",
                extra_envs_applied={},
                specialist_task_id=specialist_task_id,
                state_model_path=state_model_path,
                unset_envs=switch_names,
                variant_suffix="-parity",
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
        except Exception as exc:  # noqa: BLE001 — a failed probe must not read as a pass
            return {
                "ran": True,
                "ok": False,
                "reason": f"switch-off parity leg raised: {exc!r}",
            }
        parity_tput = parity_bench.get("output_throughput")
        accuracy_pass = parity_evidence.get("accuracy_pass")
        if not isinstance(parity_tput, (int, float)) or parity_tput <= 0:
            # No measurement is not evidence of a behavioural change. The patch is
            # still reverted — leaving an unverified rewrite on disk would skew every
            # later measurement — but the verdict must not claim the invariant was
            # tested and broken. On a live session this exact branch discarded a
            # +4.7% patch whose parity leg had in fact measured 0.5% from base, and
            # recording that as a violation would have taught later sessions a lesson
            # drawn from a filesystem race rather than from the code.
            return {
                "ran": True,
                "ok": False,
                "inconclusive": True,
                "tput": parity_tput,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity could not be measured: the parity leg returned "
                    "no throughput, so the switches-unset invariant was never tested. "
                    "Reverting because an unverified rewrite must not stay on disk, not "
                    "because the patch was shown to be non-inert"
                ),
            }
        delta_pct = (float(parity_tput) - base_tput) / base_tput * 100.0
        if abs(delta_pct) > band_pct:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    f"switch-off parity leg moved throughput {delta_pct:+.2f}% "
                    f"(band +/-{band_pct:.2f}%): the patch changes behaviour with "
                    f"every switch unset, so it is not a default-off rewrite"
                ),
            }
        if accuracy_pass is False:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity leg failed its correctness gate: the patch "
                    "changes output with every switch unset"
                ),
            }
        return {
            "ran": True,
            "ok": True,
            "tput": float(parity_tput),
            "delta_pct": delta_pct,
            "band_pct": band_pct,
            "accuracy_pass": accuracy_pass,
            "reason": f"switch-off parity within +/-{band_pct:.2f}% ({delta_pct:+.2f}%)",
        }

    async def _keep_inert_switches(
        self,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        specialist_task_id: str,
        done_payload: dict[str, Any] | None,
        output_root: Path,
        framework_root: Path | None,
        stash_state: str,
        stash_note: str,
        applied: list[Path],
        applied_artifacts: list[dict[str, Any]],
        switch_manifest: list[dict[str, Any]],
        switch_problems: list[str],
        parity: dict[str, Any],
        bench_result: dict[str, Any],
        new_tput: Any,
        delta_pct: float | None,
        base_tput: float,
        keep_threshold_pct: float,
        accuracy_pass: bool | None,
        acc_delta_pct: float | None,
        cfg_fingerprint: str,
    ) -> dict[str, Any]:
        """Keep a correct-but-unprofitable rewrite patch dormant and register its levers.

        The bundle passed correctness but not the throughput threshold. Because
        every rewrite is behind a switch that defaults OFF, the applied code is
        inert: leaving it in place costs nothing at runtime, while reverting it
        would discard the rewrites that do pay along with the one that does not,
        and would discard any enabler whose whole purpose is to make another
        rewrite profitable rather than to be profitable itself.

        So the code stays and the switches are registered as search levers, with
        ``extra_envs_applied`` deliberately empty so nothing enters the running
        configuration. The explore phase then turns them on one dependency-closed
        bundle at a time.

        Args:
            params: The task params.
            extra: The runner's extra context.
            specialist_task_id: The originating specialist.
            done_payload: The specialist's done payload, for the KB record.
            output_root: The per-task workspace.
            framework_root: The patched framework checkout.
            stash_state: Stash bookkeeping for the restore wrapper.
            stash_note: Stash bookkeeping for the restore wrapper.
            applied: Patches that were applied and are being kept.
            applied_artifacts: Artifacts that were installed.
            switch_manifest: Parsed switch manifest.
            switch_problems: Problems found while parsing it.
            parity: The switch-off parity verdict, recorded on the result so the
                inert KEEP carries its own evidence of being inert.
            bench_result: The measured bench result (switches on).
            new_tput: Measured throughput with the switches on.
            delta_pct: Measured delta against ``base_tput``.
            base_tput: The comparison base.
            keep_threshold_pct: The throughput threshold that was not met.
            accuracy_pass: Accuracy verdict.
            acc_delta_pct: Accuracy delta, for the KB record.
            cfg_fingerprint: Config fingerprint, for the KB record.

        Returns:
            The ``kept_inert`` result envelope.
        """
        enablers = [entry["switch"] for entry in switch_manifest if entry.get("enabler")]
        reason_bits = [
            f"bundle throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%"
            if delta_pct is not None
            else "bundle throughput not measurable",
            f"code kept inert ({len(applied)} patch(es), all switches default-off) and "
            f"{len(switch_manifest)} lever(s) registered for per-lever exploration",
        ]
        if enablers:
            reason_bits.append(
                f"{len(enablers)} declared enabler(s) ({', '.join(enablers)}) cannot pay standalone "
                f"and are only measurable inside their bundle"
            )
        await self._maybe_write_framework_kb_record(
            params=params,
            done_payload=done_payload,
            outcome="kept_inert_levers_registered",
            tps_delta_pct=float(delta_pct or 0.0),
            extra=extra,
            accuracy_delta_pct=acc_delta_pct,
            config_fingerprint=cfg_fingerprint,
        )
        log.info(
            "integrate_patch: KEEP_INERT task=%s delta=%s threshold=%.2f%% levers=%d enablers=%d",
            specialist_task_id,
            f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a",
            keep_threshold_pct,
            len(switch_manifest),
            len(enablers),
        )
        return _with_stash_restore(
            framework_root,
            stash_state,
            stash_note,
            {
                "status": "kept_inert",
                # True when the bundle moved the output with every switch on. The
                # code is still kept, because the switches are benched together and
                # that verdict does not say which one is at fault — explore bisects
                # per lever from here. The flag exists so nothing downstream reads
                # this as a clean keep.
                "quality_unverified": accuracy_pass is False,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "artifacts_applied": applied_artifacts,
                # Empty on purpose: the code is present but dormant, so nothing
                # may enter current_best. The levers below are how it gets turned
                # on, one measured bundle at a time.
                "config_changes_applied": {},
                "extra_server_args_applied": "",
                "extra_envs_applied": {},
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": "; ".join(reason_bits),
                "bench_result": bench_result,
                "workspace": str(output_root),
                "framework_root": str(framework_root or ""),
                "framework_levers": switch_manifest,
                "framework_lever_outcome": "registered_off",
                "framework_switch_problems": switch_problems,
                "switch_off_parity": parity,
            },
        )

    # Helpers
    @staticmethod
    def _find_frameworkoposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.
            params: Task params; the upstream-PR lane's PR identity lives on
                ``params['candidate']`` rather than in a specialist proposal.

        Returns:
            The matching framework proposal dict, or ``None`` when absent.
        """
        if not isinstance(done_payload, dict):
            return None
        proposal_set = done_payload.get("proposal_set") or []
        if not isinstance(proposal_set, list):
            return None
        for proposal in proposal_set:
            if not isinstance(proposal, dict):
                continue
            provenance = str(proposal.get("provenance") or "")
            if provenance.startswith("specialist:serving:framework"):
                return proposal
        return None

    @staticmethod
    def _upstream_pr_kb_proposal(params: Mapping[str, Any]) -> dict[str, Any] | None:
        """Present an upstream-PR candidate in the shape the KB writer reads.

        The ``fa_pr_url`` / ``fa_pr_sha`` keys exist because the PR identity used
        to reach this executor only by being smuggled through a specialist's
        output. A candidate row carries it directly, so map rather than relay.

        Args:
            params: Task params, read for ``candidate``.

        Returns:
            A proposal-shaped mapping, or ``None`` when this is not an
            upstream-PR task or the candidate carries no dedup key.
        """
        if resolve_patch_source(params) != PATCH_SOURCE_UPSTREAM_PR:
            return None
        candidate = params.get("candidate")
        if not isinstance(candidate, dict):
            return None
        pr_url = str(candidate.get("pr_url") or candidate.get("url") or "").strip()
        pr_sha = str(candidate.get("head_sha") or "").strip()
        if not pr_url and not pr_sha:
            return None
        return {
            "fa_pr_url": pr_url,
            "fa_pr_sha": pr_sha,
            "framework": candidate.get("framework") or "",
            "gap_canonical_id": candidate.get("gap_canonical_id") or "",
            "gap_keywords": candidate.get("gap_keywords") or [],
            "changed_files": candidate.get("changed_files") or [],
            "applicability": candidate.get("applicability") or "",
            # Names the lever, matching ``lever_kind``. No ledger reader filters
            # on provenance; it is audit metadata.
            "provenance": LEVER_UPSTREAM_PR,
            "source_framework": candidate.get("source_framework") or "",
            "target_framework": candidate.get("target_framework") or "",
        }

    async def _maybe_write_framework_kb_record(
        self,
        *,
        done_payload: dict[str, Any] | None,
        params: Mapping[str, Any] | None = None,
        outcome: str,
        tps_delta_pct: float,
        extra: dict[str, Any],
        accuracy_delta_pct: float | None = None,
        config_fingerprint: str = "",
    ) -> None:
        """Append a JSONL record to ``lessons.jsonl`` when the patch
        carries an upstream PR identity.

        No-op for other provenance or when both dedup keys (``fa_pr_url`` /
        ``fa_pr_sha``) are missing. Write errors are logged + swallowed.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.
            outcome: The outcome label to record (e.g. integrated / reverted).
            tps_delta_pct: The measured throughput delta percentage.
            extra: The runner ``extra`` mapping (provides shared state /
                session id).
            accuracy_delta_pct: Measured accuracy delta; overrides the payload
                value when supplied.
            config_fingerprint: Content fingerprint of the applied server
                args / envs, recorded so a retried config can be recognised.
        """
        proposal = self._find_frameworkoposal(done_payload)
        if proposal is None:
            # The upstream-PR lane carries the PR identity on the candidate
            # rather than in a specialist's ``fa_*`` markers. Discovery dedups
            # on this ledger, so a lane that writes none is re-benched forever.
            proposal = self._upstream_pr_kb_proposal(params or {})
            if proposal is None:
                return
        pr_url = str(proposal.get("fa_pr_url") or "").strip()
        pr_sha = str(proposal.get("fa_pr_sha") or "").strip()
        if not pr_url and not pr_sha:
            log.warning(
                "integrate_patch: framework proposal lacks both fa_pr_url and fa_pr_sha; KB writeback skipped",
            )
            return
        patches_written = proposal.get("patches_written") or []
        patch_path = ""
        if isinstance(patches_written, list) and patches_written:
            patch_path = str(patches_written[0])
        session_id = ""
        shared_state = extra.get("shared_state") or extra.get("state")
        if shared_state is not None:
            session_id = str(getattr(shared_state, "recipe_kb_session_id", "") or "")
        try:
            from ...knowledge.kb_writeback import write_framework_record

            gap_keywords = proposal.get("gap_keywords") or (done_payload or {}).get("gap_keywords") or []
            if isinstance(gap_keywords, str):
                gap_keywords = [gap_keywords]
            changed_files = proposal.get("changed_files") or (done_payload or {}).get("changed_files") or []
            if isinstance(changed_files, str):
                changed_files = [changed_files]
            if accuracy_delta_pct is None:
                try:
                    accuracy_delta_pct = float(
                        proposal.get("accuracy_delta_pct") or (done_payload or {}).get("accuracy_delta_pct") or 0.0
                    )
                except (TypeError, ValueError):
                    accuracy_delta_pct = 0.0
            written = await write_framework_record(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=patch_path,
                outcome=outcome,
                tps_delta_pct=float(tps_delta_pct),
                session_id=session_id,
                framework=str(proposal.get("framework") or (done_payload or {}).get("framework") or "").strip().lower(),
                gap_canonical_id=str(
                    proposal.get("gap_canonical_id") or (done_payload or {}).get("gap_canonical_id") or ""
                ).strip(),
                gap_keywords=[str(k).strip().lower() for k in gap_keywords if str(k).strip()],
                model_class=str(getattr(shared_state, "model_class", "") if shared_state is not None else "").strip(),
                gpu_type=str(getattr(shared_state, "gpu_type", "") if shared_state is not None else "").strip(),
                precision=str(getattr(shared_state, "precision", "") if shared_state is not None else "").strip(),
                applicability=(
                    config_fingerprint
                    or str(proposal.get("applicability") or (done_payload or {}).get("applicability") or "").strip()
                ),
                provenance=str(proposal.get("provenance") or (done_payload or {}).get("provenance") or "").strip(),
                accuracy_delta_pct=accuracy_delta_pct,
                changed_files=[str(f).strip() for f in changed_files if str(f).strip()],
                source_framework=str(
                    proposal.get("source_framework") or (done_payload or {}).get("source_framework") or ""
                )
                .strip()
                .lower(),
                target_framework=str(
                    proposal.get("target_framework") or (done_payload or {}).get("target_framework") or ""
                )
                .strip()
                .lower(),
                session_dir=self.session_dir,
            )
            log.info(
                "integrate_patch: wrote framework KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
                written,
                outcome,
                pr_url,
                float(tps_delta_pct),
            )
        except Exception as exc:  # noqa: BLE001 — KB write is best-effort
            log.warning(
                "integrate_patch: framework KB writeback failed: %r",
                exc,
            )

    def _undo_ungraded_candidate(self, ctx: Any) -> None:
        """Take the candidate back out when a stage unwound instead of returning.

        Every REVERT the stages themselves decide hangs off an ``except
        Exception``, and the stop that matters most here is not one of those: the
        dispatcher cancels in-flight actions on shutdown and on a spent
        wall-clock budget, and ``CancelledError`` derives from ``BaseException``.
        Unhandled, it leaves the patch in the framework tree and the operator's
        auto-stash on the stack — and the budget case does not end the process,
        so CLOSE would report against a tree carrying a patch nothing ever
        graded.

        The cancel itself is re-raised by the caller rather than turned into a
        REVERT verdict, so the run records it the way
        :mod:`..stop_attribution` requires: work the run stopped, not work that
        failed. Every step here is synchronous, so no second cancel can be
        delivered part-way through the undo.

        Read from ``ctx`` rather than from arguments because a stop can arrive
        mid-stage, before the stage has returned anything to the caller: what the
        undo owes is exactly what the tree has already been given, and each stage
        publishes that as it happens. A stage that has not stashed yet leaves
        ``clean`` behind, which makes the whole undo a no-op.

        Args:
            ctx: The runner context the stages publish their ``_ip_*``
                tree-mutation bookkeeping onto.
        """
        framework_root: Path | None = getattr(ctx, "_ip_framework_root", None)
        self._revert_artifacts(list(getattr(ctx, "_ip_applied_artifacts", None) or []))
        self._revert_patches(framework_root, list(getattr(ctx, "_ip_applied", None) or []))
        if framework_root is not None:
            _restore_stash_logged(
                framework_root,
                str(getattr(ctx, "_ip_stash_state", "") or "clean"),
                str(getattr(ctx, "_ip_stash_note", "") or ""),
            )

    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
    ) -> list[Path]:
        """Reverse-apply the applied patches (best-effort); returns those
        actually reverted.

        ``applied`` does not decide whether a restore is owed. On non-git trees
        the backup ledger does; on git trees a patch set that fails part-way
        through its first patch, or a round whose only write was the
        base-artifact replay, has mutated the tree while ``applied`` is empty.

        Args:
            framework_root: The source root to revert in, or ``None`` (no-op).
            applied: The patches that were applied this run.

        Returns:
            The patches actually reverted (may be the full ``applied`` list
            when the checkout fallback fires).
        """
        reverted: list[Path] = []
        if framework_root is None:
            return reverted
        nogit_backups = getattr(self, "_nogit_patch_backups", None)
        if nogit_backups is not None and not _is_git_tree(framework_root):
            if nogit_backups:
                ok, errors = _revert_patches_no_git(nogit_backups)
                if not ok:
                    log.error("integrate_patch: non-git revert incomplete in %s: %s", framework_root, errors)
                    return []
            return list(applied)
        if not self._apply_attempted and not applied and not self._ip_base_artifact_replayed:
            return reverted
        # Restoring to HEAD is the revert, not a fallback for one. Every KEEP is
        # committed, so HEAD is exactly the accepted stack: kept work is in
        # commits and survives, candidate work is uncommitted and goes. User
        # state was stashed before the apply.
        #
        # Reverse-applying each diff was the old primary path and is where the
        # residue came from: a forward apply that used fuzz or a guessed -p level
        # reverses to something a few lines off HEAD, `git apply -R` still
        # reports success, and what is left behind gets banked as user state by
        # the next candidate's auto-stash.
        ok, err = _git_checkout_clean(framework_root)
        if ok:
            return list(applied)
        log.error(
            "integrate_patch: could not restore %s to HEAD (%s); falling back to reverse-apply",
            framework_root,
            err,
        )
        # Reverse order so dependent patches unstick correctly.
        for patch in reversed(applied):
            ok_rev, err_rev = _git_apply_reverse(framework_root, patch)
            if ok_rev:
                reverted.append(patch)
            else:
                log.warning(
                    "integrate_patch: git apply -R failed for %s: %s",
                    patch,
                    err_rev,
                )
                break
        return reverted

    def _replay_base_artifacts(self, params: dict[str, Any]) -> None:
        """Re-install artifacts that prior enablement rounds accepted.

        Called twice per round: in launch-only mode (before the early return, so
        the probe boots against the accepted stack) and after the framework stash
        (so the re-install is not immediately swept up by ``_git_stash_if_dirty``).
        The replay does not feed ``applied_artifacts``; base artifacts are the
        stable base and must not be reverted when this round's candidate is rolled back.

        Each entry is validated before installation:
        - ``target`` must resolve inside an allowlisted framework root
          (via :func:`_resolve_artifact_target`).
        - ``source`` must resolve inside the session directory.

        An entry failing either check is skipped with a warning. The install
        itself is unguarded on purpose: a base artifact belongs to the accepted
        stack, so a round that cannot restore it would benchmark a tree no round
        asked for. The ``OSError`` propagates and ``__call__`` unwinds it.
        """
        if not bool(params.get("enablement")):
            return
        base_artifacts = params.get("enablement_base_artifacts")
        if not isinstance(base_artifacts, list):
            return
        for art in base_artifacts:
            if not isinstance(art, dict):
                continue
            source_str = str(art.get("source") or "").strip()
            target_str = str(art.get("target") or "").strip()
            if not source_str or not target_str:
                continue
            source = Path(source_str)
            if not source.is_file():
                log.warning("integrate_patch: base artifact source not found: %s", source)
                continue
            try:
                source_resolved = source.resolve()
            except (OSError, RuntimeError):
                log.warning("integrate_patch: base artifact source unresolvable: %s", source)
                continue
            if not _is_within(source_resolved, self.session_dir.resolve()):
                log.warning(
                    "integrate_patch: base artifact source %s escapes session dir; skipping",
                    source,
                )
                continue
            resolved = _resolve_artifact_target(target_str)
            if resolved is None:
                log.warning(
                    "integrate_patch: base artifact target %r not in allowlisted root; skipping",
                    target_str,
                )
                continue
            target = resolved[0]
            # Set before the write so a copy that raises still counts as dirtying.
            self._ip_base_artifact_replayed = True
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            log.info("integrate_patch: re-installed base artifact %s", target)

    def _apply_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Install non-diff tuned artifacts, backing up any clobbered targets.

        Each artifact's existing target is backed up under ``backup_root`` (or
        recorded as newly-created) so :meth:`_revert_artifacts` can restore the
        framework tree exactly. Applied artifacts are returned in order so a
        revert can undo them in reverse.

        Args:
            specs: The resolved artifact specs to install.
            backup_root: Directory under which clobbered targets are saved.

        Returns:
            A ``(applied, errors)`` tuple: per-artifact apply records (with the
            backup bookkeeping) and per-artifact error records.
        """
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        backup_root.mkdir(parents=True, exist_ok=True)
        for idx, spec in enumerate(specs):
            try:
                spec.target.parent.mkdir(parents=True, exist_ok=True)
                existed = spec.target.exists()
                backup_path: str | None = None
                if existed:
                    backup_path = str(backup_root / f"{idx:03d}_{spec.target.name}.bak")
                    shutil.copy2(spec.target, backup_path)
                shutil.copy2(spec.source, spec.target)
                applied.append(
                    {
                        "target": str(spec.target),
                        "rel_target": spec.rel_target,
                        "root": str(spec.root),
                        "kind": spec.kind,
                        "existed": existed,
                        "backup": backup_path,
                        # Re-install source for the next round's base replay and
                        # for the archived copy in enablement_setting.sh.
                        "source": str(spec.source),
                    }
                )
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
        return applied, errors

    @staticmethod
    def _revert_artifacts(applied: list[dict[str, Any]]) -> list[str]:
        """Undo installed artifacts (restore backups / delete created files).

        Args:
            applied: The apply records returned by :meth:`_apply_artifacts`.

        Returns:
            The framework-relative targets actually reverted.
        """
        reverted: list[str] = []
        for rec in reversed(applied):
            target = Path(str(rec.get("target") or ""))
            if not target.name:
                continue
            try:
                if rec.get("existed") and rec.get("backup"):
                    shutil.copy2(str(rec["backup"]), target)
                elif not rec.get("existed"):
                    if target.exists():
                        target.unlink()
                reverted.append(str(rec.get("rel_target") or target))
            except OSError as exc:  # noqa: BLE001 — best-effort restore
                log.warning("integrate_patch: failed to revert artifact %s: %r", target, exc)
        return reverted

    async def _bench_patch(
        self,
        *,
        params: dict[str, Any],
        output_root: Path,
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        specialist_task_id: str,
        state_model_path: str = "",
        unset_envs: "list[str] | None" = None,
        variant_suffix: str = "",
        session_deadline_sec: float | None = None,
        variant_expected_sec: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy gate.

        Args:
            params: The task params (config / model / bench knobs).
            output_root: The per-task workspace root for the bench.
            state_model_path: ``SharedState.model_path``, the last rung of the
                model-path precedence. Passed in because the caller owns the
                session context.
            extra_server_args_applied: Server CLI arguments for the variant.
            extra_envs_applied: Environment overrides layered onto the variant.
            specialist_task_id: The originating specialist task id (names the
                variant).
            unset_envs: Extra env names to remove for this leg, on top of the
                task's ``base_unset_envs``. Used by the switch-off parity leg,
                which has to guarantee the rewrite switches are absent even when
                an earlier KEEP put them into the base configuration.
            variant_suffix: Appended to the variant name so a second leg does not
                collide with the first one's grid slot.
            session_deadline_sec: Monotonic-clock session budget deadline, or
                ``None`` when unbounded. Resolved by the caller, which owns the
                session context.
            variant_expected_sec: Expected bench runtime used to decide whether
                the remaining budget can fit this bench at all.

        Returns:
            A ``(bench_result_dict, gate_evidence)`` tuple where
            ``bench_result_dict`` carries ``effective_config`` (the env/arg layers
            the variant launched with, for a faithful replay), and
            ``gate_evidence`` carries ``accuracy_pass`` (True / False / None)
            and ``eval_probe`` (the generation-pathology record, or ``None``).
        """
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            raise RuntimeError(f"integrate_patch bench: config not found at {config_path}")
        # Last rung of the precedence: without it a task with no params
        # model_path and no MODEL_PATH in env resolves to the empty string.
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=state_model_path,
            for_serving=True,
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            extra_envs=self._framework_run_eval_envs(params),
            remove_args=params.get("base_remove_args"),
            unset_envs=params.get("base_unset_envs"),
            args_mode=str(params.get("base_args_mode") or "append"),
            out_name="integrate_patch.with_envs.yaml",
        )

        _base_envs = dict(params.get("base_extra_envs") or {})
        _variant_envs = dict(_base_envs)
        _variant_envs.update(extra_envs_applied)
        # Eval-origin enablement needs a raw accuracy for its runnable gate, so
        # RUN_EVAL=true must survive any variant overlay.
        if bool(params.get("enablement")) and _is_eval_origin(params):
            _variant_envs["RUN_EVAL"] = "true"
        _unset = to_str_list(params.get("base_unset_envs"))
        for name in unset_envs or []:
            key = str(name).strip()
            if not key:
                continue
            # Removing it from the variant env too: ``unset_envs`` drops inherited
            # values, but a key present in both would otherwise be re-added here.
            _variant_envs.pop(key, None)
            if key not in _unset:
                _unset.append(key)
        variant = GridVariant(
            name=f"integrate-patch-{specialist_task_id[:8]}{variant_suffix}",
            extra_server_args=extra_server_args_applied,
            extra_envs=_variant_envs,
            remove_args=to_str_list(params.get("base_remove_args")),
            unset_envs=_unset,
            args_mode=str(params.get("base_args_mode") or "append"),
            note=f"integrate_patch:{specialist_task_id}{variant_suffix}",
        )
        _rt = params.get("runtime_override")
        if isinstance(_rt, dict) and _rt:
            # Preserve list/dict values; apply_runtime_override expects them.
            variant.runtime_override = dict(_rt)

        # Ray-managed GPU execution: hold a serving lease
        # (num_gpus=TP + serving_slot) for the whole run_grid so
        # the patch benchmark serializes against other serving on the
        # whole-machine mutex instead of colliding with a concurrently-running
        # GPU specialist server on the same card (the observed
        # ``reverted_smoke_fail`` root cause). ``None`` keeps the local path
        # (multi-node / RAY_EXEC off / pytest default).
        from ._ray_serving import maybe_serving_lease

        serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
            results: list[VariantResult] = await run_grid(
                base_yaml_path=config_path,
                base_extra_args=str(params.get("base_extra_args") or "").strip(),
                grid=[variant],
                output_root=output_root,
                magpie_python=params.get("magpie_python") or None,
                variant_timeout_sec=int(
                    params.get("variant_timeout_sec", self.variant_timeout_sec),
                ),
                keep_going_on_failure=False,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                result_dir=override_result_dir,
                base_args_mode=str(params.get("base_args_mode") or "append"),
                serving_lease=serving_lease,
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
        finally:
            if serving_lease is not None:
                serving_lease.close()

        bench: dict[str, Any] = {}
        if results:
            r = results[0]
            bench = {
                "name": r.name,
                "status": r.status,
                "output_throughput": getattr(r, "output_throughput", None),
                # ``VariantResult`` names these ``ttft_mean_ms`` / ``tpot_mean_ms``;
                # the emitted keys stay ``ttft_ms`` / ``itl_ms`` for the collectors.
                "ttft_ms": r.ttft_mean_ms,
                "itl_ms": r.tpot_mean_ms,
                # Benchmark dir; ``_grade_accuracy`` locates accuracy artifacts here.
                "workspace": str(getattr(r, "workspace", "") or ""),
                "error": getattr(r, "error", "") or "",
                "error_class": getattr(r, "error_class", "") or "",
                "nonfatal_warnings": list(getattr(r, "nonfatal_warnings", []) or []),
                # Launch evidence is the immutable proof of the server that
                # produced this measurement. It must survive the patch result
                # and current-best promotion so GEAK can verify the handoff.
                "launch_evidence": dict(getattr(r, "launch_evidence", {}) or {}),
                "launch_evidence_path": str(getattr(r, "launch_evidence_path", "") or ""),
                "server_log_path": str(getattr(r, "server_log_path", "") or ""),
                # Materialized config used for this bench; needed by revalidation.
                "materialized_config": str(config_path),
                # Read off the variant so a replay cannot drift from the graded run.
                # RUN_EVAL is dropped: the replay owns its own eval contract.
                "effective_config": {
                    "extra_envs": {k: v for k, v in variant.extra_envs.items() if k != "RUN_EVAL"},
                    "extra_server_args": compose_server_args(
                        inherited_args="",
                        base_extra_args=str(params.get("base_extra_args") or "").strip(),
                        variant_extra_args=variant.extra_server_args,
                        remove_args=variant.remove_args,
                        args_mode=variant.args_mode,
                    ),
                    "remove_args": list(variant.remove_args),
                    "unset_envs": list(variant.unset_envs),
                    "args_mode": variant.args_mode,
                },
            }

        accuracy_pass: bool | None = None
        # lm-eval writes to ``$EVAL_RESULT_DIR`` under the grid slot, not inside
        # the ``benchmark_*`` workspace. Grade from the slot so the recursive
        # search finds eval output while honoring an explicit ``result_dir``
        # override the same way the grid subprocess does.
        eval_search_root = override_result_dir or (
            str(Path(bench["workspace"]).parent) if bench.get("workspace") else ""
        )
        if bench.get("status") == "succeeded":
            accuracy_pass = self._grade_accuracy(
                eval_search_root,
                params.get("accuracy_baseline"),
                framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
            )

        # Raw accuracy for the KB record; ``accuracy_pass`` only carries a verdict.
        measured_accuracy: float | None = None
        if bench.get("status") == "succeeded":
            try:
                measured = parse_eval_results(
                    eval_search_root,
                    framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
                ).get("accuracy")
                if isinstance(measured, (int, float)):
                    measured_accuracy = float(measured)
            except Exception:  # noqa: BLE001 — advisory value only
                log.debug("integrate_patch: accuracy parse for KB record failed", exc_info=True)

        # Enablement path: surface the raw accuracy so the branch can apply a floor.
        enablement_accuracy: float | None = None
        enablement_accuracy_task = ""
        enablement_accuracy_metric = ""
        if bool(params.get("enablement")) and bench.get("status") == "succeeded":
            try:
                eval_results = parse_eval_results(
                    eval_search_root,
                    framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
                )
                acc = eval_results.get("accuracy")
                if isinstance(acc, (int, float)):
                    enablement_accuracy = float(acc)
                enablement_accuracy_task = str(eval_results.get("task") or "")
                enablement_accuracy_metric = str(eval_results.get("metric") or "")
            except Exception:  # noqa: BLE001 — eval may not produce a result
                log.debug("integrate_patch: enablement eval parse failed", exc_info=True)

        # Guarded: an empty root would send the recursive scan over the cwd.
        eval_probe = read_eval_probe(eval_search_root) if eval_search_root else None

        return bench, {
            "accuracy_pass": accuracy_pass,
            "accuracy": measured_accuracy,
            "enablement_accuracy": enablement_accuracy,
            "enablement_accuracy_task": enablement_accuracy_task,
            "enablement_accuracy_metric": enablement_accuracy_metric,
            "eval_probe": eval_probe,
        }

    @staticmethod
    def _framework_run_eval_envs(params: dict[str, Any]) -> dict[str, Any] | None:
        """Force ``RUN_EVAL=true`` for framework-authored source patches.

        Two independent triggers:

        * **Eval-origin enablement**: force ``RUN_EVAL=true`` so ``_bench_patch``
          can obtain a raw accuracy for the runnable gate, which fails closed
          without one. A boot-origin candidate is only ever provisional on a
          missing accuracy, so it inherits the session's contract instead.
        * **Perf framework authoring**: force only when a comparable baseline
          accuracy exists (``accuracy_baseline > 0``); otherwise leave the
          candidate's ``RUN_EVAL`` to the materializer's default handling.

        A plain configuration integrate_patch is untouched (returns ``None``).

        Args:
            params: The integrate_patch task params.

        Returns:
            ``{"RUN_EVAL": "false"}`` when the session disabled evals,
            ``{"RUN_EVAL": "true"}`` for eval-origin enablement patches or for
            framework-authored perf patches with a positive baseline accuracy
            to compare against; else ``None``.
        """
        # The session's opt-out outranks every force-on below.
        if is_truthy(params.get("disable_run_eval")):
            return {"RUN_EVAL": "false"}
        if bool(params.get("enablement")):
            return {"RUN_EVAL": "true"} if _is_eval_origin(params) else None
        fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
        try:
            baseline = float(params.get("accuracy_baseline") or 0.0)
        except (TypeError, ValueError):
            baseline = 0.0
        return {"RUN_EVAL": "true"} if (fw_authored and baseline > 0) else None

    @staticmethod
    def _grade_accuracy(
        result_dir: str,
        baseline_accuracy: Any,
        framework: str | None = None,
    ) -> bool | None:
        """Grade a bench's accuracy against the baseline.

        With a recorded baseline the measured drop is enforced; without one
        (or no eval result) the check is skipped (``None``) and warned loudly.
        For scriptable frameworks (xDiT) ``parse_eval_results`` fails closed on
        a missing quality gate instead of falling back to GSM8K.
        """
        # Accept numeric strings in addition to int/float; non-numeric / missing
        # values fall back to 0.0 (skip).
        try:
            baseline_value = float(baseline_accuracy)
        except (TypeError, ValueError):
            baseline_value = 0.0
        try:
            eval_results = parse_eval_results(result_dir, framework=framework)
            new_accuracy = eval_results.get("accuracy")
            if new_accuracy is not None and baseline_value > 0:
                return accuracy_passed(baseline_value, float(new_accuracy))
            if baseline_value <= 0:
                log.warning(
                    "integrate_patch: no baseline accuracy; accuracy gate skipped "
                    "(throughput-only KEEP). Accuracy regressions will not be caught.",
                )
            else:
                log.warning("integrate_patch: variant produced no accuracy result; gate skipped")
        except Exception:  # noqa: BLE001
            log.exception("integrate_patch: accuracy gate parse failed; treating as None (gate skipped)")
        return None


__all__ = [
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_VARIANT_TIMEOUT_SEC",
    "IntegratePatchExecutor",
    "_detect_p_level",
    "_git_apply",
    "_git_apply_reverse",
    "_git_checkout_clean",
    "_run_git_apply",
    "_resolve_framework_root",
    "_resolve_patch_paths",
    "_read_done_payload",
]
