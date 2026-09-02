#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run KernelForge against a traced multi-GPU collective kernel."""

from __future__ import annotations

import argparse
import base64
import binascii
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _invocation_spec import (  # noqa: E402
    build_invocation_spec,
    invocation_spec_filename,
    write_invocation_spec,
)
from collective_driver_generator import (  # noqa: E402
    SNR_FLOOR_DB,
    generate_collective_driver,
)

sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
from _llm_stability_env import apply_llm_stability_env  # noqa: E402
from forge_submit import (  # noqa: E402
    _acquire_repo_lock,
    _export_best_artifacts,
    _needs_inplace,
    _new_forge_branch,
    _prepare_inplace,
    _prepare_worktree,
    _read_forge_best_result,
    _release_repo_lock,
    _remove_worktree,
    _restore_inplace,
    _terminate_forge_process,
    _untracked_paths,
    _validated_forge_best_result,
)


sys.path.pop(0)

log = logging.getLogger(__name__)

RESULT_BEGIN = "FORGE_COLLECTIVE_RESULT_BEGIN"
RESULT_END = "FORGE_COLLECTIVE_RESULT_END"
DEFAULT_TIMEOUT_SEC = 14400  # 4h: a collective iterates over N ranks per bench.
DEFAULT_SNR_THRESHOLD = SNR_FLOOR_DB
DEFAULT_FINALIZE_GRACE_SEC = 300
MIN_CAMPAIGN_TIMEOUT_SEC = 60
#: Caller-owned campaign id. Pinning it makes forge-loop's checkpoint filename
#: predictable, which is what external recovery reads after a hard kill.
EXPERIMENT_ID = "hyperloom_collective"
#: aiter implements every collective this lane can reach -- all-reduce,
#: reduce-scatter and all-gather all live in its custom_all_reduce sources -- so
#: its kernel backend (and the matching knowledge base) is the correct specialist.
COLLECTIVE_KERNEL_BACKEND = "aiter"
FORGE_SHUTDOWN_GRACE_SEC = 30


def _inject_author_gateway_env() -> None:
    """Seed missing Forge author credentials and Git identity."""
    openai_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if openai_base and not os.environ.get("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = openai_base[:-3] if openai_base.endswith("/v1") else openai_base
    token = os.environ.get("OPENAI_API_KEY", "").strip()
    if token:
        os.environ.setdefault("ANTHROPIC_API_KEY", token)
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", token)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    os.environ.setdefault("GIT_AUTHOR_NAME", "forge-bot")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "forge-bot@local")
    os.environ.setdefault("GIT_COMMITTER_NAME", "forge-bot")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "forge-bot@local")
    apply_llm_stability_env(os.environ)


def _load_input_json(path: str) -> dict[str, Any]:
    """Load the wrapper input object."""
    if not path:
        raise ValueError("--input-json is required")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {path}")
    return data


def _add_opt(cmd: list[str], value: Any, flag: str) -> None:
    """Append one populated CLI option."""
    if value not in (None, "", []):
        cmd.extend([flag, str(value)])


class CollectiveInvocationSpecUnavailable(RuntimeError):
    """The evidence forge-loop needs to author ``run_candidate`` is missing."""


def _write_invocation_evidence(candidate: dict[str, Any], output_dir: Path) -> str:
    """Record how the traced collective is called, for forge-loop task prep.

    The spec carries the argument order, dtypes and case ids that the generated
    driver leaves as ``NotImplementedError`` for the author to fill in. Starting
    a multi-hour campaign without it only defers the failure to task
    preparation, so this raises rather than degrading -- the same call the
    rewrite lane makes when its own spec is absent.
    """
    try:
        path = output_dir / invocation_spec_filename(candidate)
        write_invocation_spec(
            path,
            build_invocation_spec(
                candidate,
                source_file=str(candidate.get("source_file") or ""),
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CollectiveInvocationSpecUnavailable(
            f"cannot record how {candidate.get('source_function') or 'the collective'} is invoked: {exc}"
        ) from exc
    return str(path)


def _campaign_is_resumable(workspace: str) -> bool:
    """Return whether the workspace holds a forge-loop campaign to continue.

    ``run_state.json`` is the only safe trigger: forge-loop requires it for
    ``--resume`` and refuses a fresh campaign over leftover artifacts.
    """
    return (Path(workspace) / "forge_experiments" / "run_state.json").is_file()


def _build_cmd(
    args: dict[str, Any],
    rig: dict[str, str],
    output_dir: Path,
    *,
    deadline_unix: int,
) -> list[str]:
    """Assemble the repository-mode ``forge-loop`` invocation."""
    source_file = str(args.get("source_file") or "").strip()
    workspace = str(args.get("kernel_repo") or "").strip()
    if not source_file:
        raise ValueError("source_file is required")
    if not workspace:
        raise ValueError("kernel_repo is required")

    resuming = _campaign_is_resumable(workspace)
    cli = args.get("cli")
    cmd = [str(cli), "forge-loop"] if cli else [sys.executable, "-m", "kernelforge.cli", "forge-loop"]
    _add_opt(cmd, workspace, "--workspace")
    if resuming:
        # forge-loop owns the campaign's immutable configuration once it has
        # been saved, and rejects any of --kernel / --driver / --program-md-file
        # / --source-files / --operator-name alongside --resume.
        cmd.append("--resume")
    else:
        _add_opt(cmd, source_file, "--kernel")
        _add_opt(cmd, rig["driver"], "--driver")
        _add_opt(cmd, rig["program"], "--program-md-file")
    _add_opt(cmd, "repository", "--task-type")
    _add_opt(cmd, args.get("git_branch"), "--git-branch")
    _add_opt(cmd, rig["world_size"], "--nproc-per-node")
    snr_threshold = args.get("snr_threshold")
    if snr_threshold in (None, ""):
        snr_threshold = DEFAULT_SNR_THRESHOLD
    if (
        isinstance(snr_threshold, bool)
        or not isinstance(snr_threshold, (int, float))
        or not math.isfinite(float(snr_threshold))
    ):
        raise ValueError("snr_threshold must be finite")
    _add_opt(cmd, snr_threshold, "--snr-threshold")
    _add_opt(cmd, args.get("gpu_target"), "--gpu-target")
    _add_opt(cmd, COLLECTIVE_KERNEL_BACKEND, "--kernel-backend")
    _add_opt(cmd, args.get("max_hours"), "--max-hours")
    if isinstance(deadline_unix, bool) or not isinstance(deadline_unix, int) or deadline_unix <= 0:
        raise ValueError("deadline_unix must be a positive integer")
    _add_opt(cmd, deadline_unix, "--deadline-unix")
    _add_opt(cmd, args.get("agent_timeout_sec"), "--agent-timeout-sec")
    # Match forge_submit / forge-fusion: pin the provider when the coordinator
    # resolved one, and disable the silent Claude fallback for Codex so a
    # missing SDK fails here instead of degrading into an unauthenticated run.
    agent_backend = str(args.get("agent_backend") or "").strip().lower()
    if agent_backend == "codex":
        cmd.extend(["--agent-backend", "codex", "--agent-fallback-provider", "none"])
    elif agent_backend == "claude":
        cmd.extend(["--agent-backend", "claude"])
    _add_opt(cmd, args.get("llm_model"), "--model")
    _add_opt(cmd, str(output_dir / "forge_result.json"), "--result-json")
    _add_opt(cmd, str(output_dir / "experiments"), "--experiments-dir")
    _add_opt(cmd, EXPERIMENT_ID, "--experiment-id")
    _add_opt(cmd, args.get("experience_id") or output_dir.name, "--experience-id")
    bench_repeat = args.get("bench_repeat")
    if bench_repeat in (None, ""):
        bench_repeat = 3
    if isinstance(bench_repeat, bool) or not isinstance(bench_repeat, int) or bench_repeat <= 0:
        raise ValueError("bench_repeat must be a positive integer")
    _add_opt(cmd, bench_repeat, "--bench-repeat")
    target_functions = args.get("target_functions")
    if not isinstance(target_functions, list) or not target_functions:
        raise ValueError("target_functions must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in target_functions):
        raise ValueError("target_functions must contain non-empty strings")
    _add_opt(
        cmd,
        ",".join(item.strip() for item in target_functions),
        "--target-functions",
    )
    if not resuming:
        source_files = args.get("source_files")
        if isinstance(source_files, list):
            joined = ",".join(item.strip() for item in source_files if isinstance(item, str) and item.strip())
            _add_opt(cmd, joined, "--source-files")
        _add_opt(cmd, args.get("operator_name"), "--operator-name")
    _add_opt(cmd, args.get("framework"), "--framework")
    spec_file = str(args.get("invocation_spec_file") or "").strip()
    if spec_file and Path(spec_file).is_file():
        _add_opt(cmd, str(Path(spec_file).resolve()), "--invocation-spec-file")
    # Gate integrity depends on task preparation pinning the driver digest;
    # disabling preparation would remove it without any local symptom.
    if "--no-prepare-task" in cmd:
        raise ValueError("collective driver integrity requires forge-loop task preparation")
    return cmd


def _timeout_sec(args: dict[str, Any]) -> int:
    """Return the validated wrapper timeout."""
    raw = args.get("timeout")
    if raw in (None, ""):
        raw = args.get("timeout_sec")
    if raw in (None, ""):
        raw = os.environ.get("FORGE_COLLECTIVE_TIMEOUT")
    if raw in (None, ""):
        raw = DEFAULT_TIMEOUT_SEC
    try:
        if isinstance(raw, bool) or isinstance(raw, float):
            raise ValueError
        timeout = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid collective timeout: {raw!r}") from exc
    if timeout <= 0:
        raise ValueError(f"collective timeout must be positive: {timeout}")
    return timeout


def _campaign_timeout_sec(args: dict[str, Any], elapsed_sec: float) -> int:
    """Reserve wrapper time for result export and repository restoration."""
    total = _timeout_sec(args)
    finalize_raw = args.get("finalize_grace_sec")
    if finalize_raw in (None, ""):
        finalize_raw = DEFAULT_FINALIZE_GRACE_SEC
    try:
        if isinstance(finalize_raw, bool) or isinstance(finalize_raw, float):
            raise ValueError
        finalize = int(finalize_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid finalize_grace_sec: {finalize_raw!r}") from exc
    if finalize < 0:
        raise ValueError("finalize_grace_sec cannot be negative")
    if (
        isinstance(elapsed_sec, bool)
        or not isinstance(elapsed_sec, (int, float))
        or not math.isfinite(float(elapsed_sec))
        or elapsed_sec < 0
    ):
        raise ValueError("elapsed_sec must be finite and non-negative")
    spent = max(0, int(elapsed_sec))
    available = total - spent - finalize
    if available < MIN_CAMPAIGN_TIMEOUT_SEC:
        raise ValueError("collective wrapper budget leaves no campaign time")
    return available


def _run_with_tree_timeout(cmd: list[str], timeout_sec: int) -> subprocess.CompletedProcess:
    """Run forge-loop in its own process group and reap the group on timeout."""
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, int) or timeout_sec <= 0:
        raise ValueError("timeout_sec must be a positive integer")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **({"start_new_session": True} if os.name == "posix" else {}),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_forge_process(
            proc,
            grace_sec=30,
        )
        raise subprocess.TimeoutExpired(cmd, timeout_sec, output=stdout, stderr=stderr)


def _persist_logs(output_dir: str, stdout: str | None, stderr: str | None) -> None:
    """Persist Forge output beside the wrapper result."""
    base = Path(output_dir or "")
    if not base.is_dir():
        return
    for name, text in (
        ("forge_loop_stdout.log", stdout),
        ("forge_loop_stderr.log", stderr),
    ):
        if not text:
            continue
        try:
            (base / name).write_text(text)
        except OSError as exc:
            print(
                f"warning: failed to persist {base / name}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    """Run a bounded git command without mutating process-global configuration."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _config_snapshot(repo: str) -> dict[str, str | None]:
    """Capture local Git identity values changed by Forge workspace helpers."""
    out: dict[str, str | None] = {}
    for key in ("user.name", "user.email"):
        proc = _git(repo, "config", "--local", "--get", key)
        if proc.returncode == 0:
            out[key] = proc.stdout.rstrip("\n")
        elif proc.returncode == 1:
            out[key] = None
        else:
            raise RuntimeError(f"could not read local Git config {key}")
    return out


def _restore_config(repo: str, snapshot: dict[str, str | None]) -> None:
    """Restore local Git identity exactly to its pre-campaign state."""
    for key, value in snapshot.items():
        if value is None:
            proc = _git(repo, "config", "--local", "--unset-all", key)
            if proc.returncode not in {0, 5}:
                raise RuntimeError(f"could not clear local Git config {key}")
        else:
            proc = _git(repo, "config", "--local", key, value)
            if proc.returncode != 0:
                raise RuntimeError(f"could not restore local Git config {key}")


def _restore_journal_path(repo: str) -> Path:
    """Return the crash-recovery journal path for an in-place campaign.

    This does not duplicate forge-loop's ``run_state.json``: that recovers
    iteration progress inside a campaign, while forge-loop explicitly leaves
    worktree prep, export and restore to its caller. Only this journal can put
    an in-place checkout back the way the operator left it.
    """
    return Path(repo) / ".git" / "hyperloom_collective_restore.json"


def _tracked_baseline_patch(repo: str) -> bytes:
    """Capture all tracked working-tree changes relative to HEAD."""
    proc = _git(
        repo,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot capture tracked repository baseline: {repo}")
    return proc.stdout.encode("utf-8")


def _write_restore_journal(repo: str, restore: dict[str, Any]) -> None:
    """Persist enough repository baseline metadata to recover after a hard crash."""
    backup = restore.get("backup")
    tracked_patch = restore.get("baseline_tracked_patch")
    if tracked_patch is None:
        tracked_patch = b""
    if not isinstance(tracked_patch, bytes):
        raise RuntimeError("collective tracked baseline patch must be bytes")
    payload = {
        key: restore.get(key)
        for key in (
            "repo",
            "orig_branch",
            "orig_head",
            "branch",
            "source_file",
            "relpath",
            "base_commit",
            "config_snapshot",
            "baseline_untracked",
            "baseline_in_base_commit",
        )
    }
    payload["backup_b64"] = base64.b64encode(backup).decode("ascii") if isinstance(backup, bytes) else ""
    payload["baseline_tracked_patch_b64"] = base64.b64encode(tracked_patch).decode("ascii")
    payload["baseline_tracked_sha256"] = hashlib.sha256(tracked_patch).hexdigest()
    path = _restore_journal_path(repo)
    tmp = path.with_suffix(".tmp")
    # The journal is the only record that can undo an in-place campaign, so it
    # has to survive a power loss, not just a process crash: fsync the contents
    # before the rename and the directory after it.
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _recorded_tree_matches(repo: str, payload: dict[str, Any]) -> bool:
    """Return whether a repository matches the journaled tree baseline."""
    expected_branch = str(payload.get("orig_branch") or "")
    expected_head = str(payload.get("orig_head") or "")
    expected_untracked = payload.get("baseline_untracked")
    expected_tracked_sha = str(payload.get("baseline_tracked_sha256") or "")
    in_memory_patch = payload.get("baseline_tracked_patch")
    if not expected_tracked_sha and isinstance(in_memory_patch, bytes):
        expected_tracked_sha = hashlib.sha256(in_memory_patch).hexdigest()
    if not isinstance(expected_untracked, list) or any(
        not isinstance(path, str) or not path for path in expected_untracked
    ):
        raise RuntimeError("collective restore journal has invalid untracked baseline")
    actual_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    actual_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if expected_tracked_sha:
        actual_tracked_sha = hashlib.sha256(_tracked_baseline_patch(repo)).hexdigest()
        tracked_matches = actual_tracked_sha == expected_tracked_sha
    else:
        tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
        tracked_matches = tracked.returncode == 0 and not tracked.stdout.strip()
    return not (
        not expected_branch
        or not expected_head
        or actual_branch != expected_branch
        or actual_head != expected_head
        or not tracked_matches
        or _untracked_paths(repo) != set(expected_untracked)
    )


def _verify_restored_repo(repo: str, payload: dict[str, Any]) -> None:
    """Verify that repository recovery restored its recorded baseline."""
    if not _recorded_tree_matches(repo, payload):
        raise RuntimeError("collective campaign did not restore its recorded baseline")


def _recover_stale_inplace(repo: str) -> bool:
    """Recover a journaled in-place campaign under the repository lock.

    The journal decides, not the branch name: an interrupted restore leaves HEAD
    back on the original branch, so a surviving journal whose tree diverged means
    the previous restore never finished and must be replayed. Replaying puts back
    the operator's pre-campaign content, including uncommitted work, and discards
    only the agent's edits.
    """
    journal = _restore_journal_path(repo)
    lock = _acquire_repo_lock(repo)
    if lock is None:
        return False
    branch_proc = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_proc.returncode != 0:
        _release_repo_lock(lock)
        raise RuntimeError(f"cannot read current branch for {repo}")
    branch = branch_proc.stdout.strip()
    parked_on_forge_branch = branch.startswith("forge/") or branch == "forge-collective-opt"
    try:
        if not journal.is_file():
            if parked_on_forge_branch:
                raise RuntimeError(f"stale Forge branch has no restore journal: {branch}")
            return True
        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"invalid collective restore journal: {journal}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"collective restore journal must be an object: {journal}")
        journal_branch = str(payload.get("branch") or "")
        if parked_on_forge_branch and journal_branch != branch:
            raise RuntimeError(f"restore journal branch {journal_branch!r} does not match {branch!r}")
        backup_encoded = payload.get("backup_b64")
        if not isinstance(backup_encoded, str):
            raise RuntimeError(f"restore journal has no source backup: {journal}")
        try:
            backup = base64.b64decode(backup_encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(f"invalid source backup in {journal}") from exc
        tracked_patch_encoded = payload.get("baseline_tracked_patch_b64", "")
        if not isinstance(tracked_patch_encoded, str):
            raise RuntimeError(f"restore journal has invalid tracked baseline: {journal}")
        try:
            tracked_patch = base64.b64decode(
                tracked_patch_encoded,
                validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(f"invalid tracked baseline in {journal}") from exc
        expected_tracked_sha = str(payload.get("baseline_tracked_sha256") or "")
        if expected_tracked_sha and hashlib.sha256(tracked_patch).hexdigest() != expected_tracked_sha:
            raise RuntimeError(f"tracked baseline checksum mismatch in {journal}")
        restore = dict(payload)
        restore["backup"] = backup
        restore["baseline_tracked_patch"] = tracked_patch
        if not parked_on_forge_branch:
            head = _git(repo, "rev-parse", "HEAD").stdout.strip()
            if branch != str(payload.get("orig_branch") or "") or head != str(payload.get("orig_head") or ""):
                # The repository advanced past the recorded baseline, so the
                # user has replaced the state this journal describes. Restoring
                # would roll their commits back.
                log.warning(
                    "Discarding a superseded collective restore journal: %s",
                    journal,
                )
                journal.unlink()
                return True
            if _recorded_tree_matches(repo, payload):
                # Still at the baseline, so the previous restore completed and
                # only the journal outlived it.
                _restore_config(repo, dict(payload.get("config_snapshot") or {}))
                _verify_restored_repo(repo, payload)
                journal.unlink()
                return True
            # HEAD never moved but the tree diverges: the previous restore was
            # interrupted after it reset HEAD and before it replayed the
            # baseline. This is indistinguishable from an ordinary branch by
            # name alone, which is why the journal decides.
        log.warning(
            "Replaying an unfinished collective restore for %s (branch=%s)",
            repo,
            branch,
        )
        restore["lock_fd"] = lock
        lock = None
        _restore_inplace(restore)
        _restore_config(repo, dict(payload.get("config_snapshot") or {}))
        _verify_restored_repo(repo, payload)
        journal.unlink(missing_ok=True)
        return True
    finally:
        _release_repo_lock(lock)


def _preserve_campaign(workspace: str, output_dir: Path, name: str) -> None:
    """Move live Forge campaign state into the session before reuse or restore."""
    campaign = Path(workspace) / "forge_experiments"
    if not campaign.exists():
        return
    lock_path = campaign / "workspace.lock"
    lock_file = lock_path.open("a+")
    try:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(f"cannot preserve active Forge campaign: {campaign}") from exc
        destination = output_dir / name
        if destination.exists():
            destination = output_dir / f"{name}_{time.time_ns()}"
        shutil.move(str(campaign), str(destination))
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _remove_verified_worktree(
    kernel_repo: str,
    source_file: str,
    workspace: str,
    branch: str,
) -> None:
    """Remove a temporary worktree and verify its branch is gone."""
    _remove_worktree(kernel_repo, source_file, workspace, branch)
    if Path(workspace).exists():
        raise RuntimeError(f"collective worktree still exists: {workspace}")
    branch_ref = _git(
        kernel_repo,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{branch}",
    )
    if branch_ref.returncode == 0:
        raise RuntimeError(f"collective branch still exists: {branch}")
    if branch_ref.returncode != 1:
        raise RuntimeError(f"cannot verify collective branch removal: {branch}")


def _prepare_collective_workspace(
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Prepare an isolated or safely restorable repository for one campaign."""
    source_file = str(payload.get("source_file") or "").strip()
    kernel_repo = str(payload.get("kernel_repo") or "").strip()
    if not source_file:
        raise ValueError("source_file is required")
    if not kernel_repo:
        raise ValueError("kernel_repo is required")

    branch = _new_forge_branch(output_dir, source_file)
    inplace = bool(_needs_inplace(kernel_repo))
    config_snapshot: dict[str, str | None]
    if inplace:
        if not _recover_stale_inplace(kernel_repo):
            return None
        lock = _acquire_repo_lock(kernel_repo)
        if lock is None:
            return None
        try:
            config_snapshot = _config_snapshot(kernel_repo)
            _preserve_campaign(
                kernel_repo,
                output_dir,
                "prior_forge_experiments",
            )
            baseline_tracked_patch = _tracked_baseline_patch(kernel_repo)
            baseline_untracked = sorted(_untracked_paths(kernel_repo))
            orig_branch_proc = _git(
                kernel_repo,
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            )
            orig_head_proc = _git(kernel_repo, "rev-parse", "HEAD")
            orig_branch = orig_branch_proc.stdout.strip()
            orig_head = orig_head_proc.stdout.strip()
            if orig_branch_proc.returncode != 0 or orig_head_proc.returncode != 0 or not orig_head:
                raise RuntimeError(f"cannot resolve repository baseline: {kernel_repo}")
            backup = Path(source_file).read_bytes()
            provisional = {
                "repo": kernel_repo,
                "orig_branch": orig_branch,
                "orig_head": orig_head,
                "branch": branch,
                "source_file": source_file,
                "backup": backup,
                "relpath": str(Path(source_file).resolve().relative_to(Path(kernel_repo).resolve())),
                "base_commit": orig_head,
                "config_snapshot": config_snapshot,
                "baseline_untracked": baseline_untracked,
                "baseline_tracked_patch": baseline_tracked_patch,
                "baseline_in_base_commit": False,
            }
            _write_restore_journal(kernel_repo, provisional)
            held_lock = lock
            lock = None
            prepared = _prepare_inplace(
                source_file,
                kernel_repo,
                branch,
                lock_fd=held_lock,
            )
        except Exception:
            _recover_stale_inplace(kernel_repo)
            raise
        finally:
            _release_repo_lock(lock)
        if prepared is None:
            _recover_stale_inplace(kernel_repo)
            return None
        workspace, prepared_source, restore = prepared
        restore["config_snapshot"] = config_snapshot
        restore["baseline_untracked"] = baseline_untracked
        restore["baseline_tracked_patch"] = baseline_tracked_patch
        restore["baseline_in_base_commit"] = False
        try:
            base_commit = str(restore.get("base_commit") or "")
            if baseline_tracked_patch:
                baseline_check = _git(
                    kernel_repo,
                    "diff",
                    "--quiet",
                    base_commit,
                    "--",
                )
                if baseline_check.returncode != 0:
                    raise RuntimeError("in-place preparation did not snapshot the tracked baseline")
            restore["baseline_in_base_commit"] = True
            _write_restore_journal(kernel_repo, restore)
            _restore_config(kernel_repo, config_snapshot)
        except Exception:
            _restore_inplace(restore)
            _restore_config(kernel_repo, config_snapshot)
            _restore_journal_path(kernel_repo).unlink(missing_ok=True)
            raise
    else:
        config_snapshot = _config_snapshot(kernel_repo)
        prepared = None
        try:
            prepared = _prepare_worktree(
                source_file,
                kernel_repo,
                output_dir,
                branch,
            )
            _restore_config(kernel_repo, config_snapshot)
        except Exception:
            try:
                _remove_verified_worktree(
                    kernel_repo,
                    source_file,
                    str(prepared[0]) if prepared is not None else str(output_dir / "worktree"),
                    branch,
                )
            finally:
                _restore_config(kernel_repo, config_snapshot)
            raise
        if prepared is None:
            return None
        workspace, prepared_source, base_commit = prepared
        restore = None
    return {
        "inplace": inplace,
        "workspace": workspace,
        "prepared_source": prepared_source,
        "base_commit": base_commit,
        "restore": restore,
        "branch": branch,
        "source_file": source_file,
        "kernel_repo": kernel_repo,
        "config_snapshot": config_snapshot,
        "output_dir": str(output_dir),
    }


def _restore_collective_workspace(context: dict[str, Any]) -> None:
    """Restore the original repository after exporting the best Forge state."""
    if context.get("inplace"):
        preserve_error: Exception | None = None
        try:
            _preserve_campaign(
                str(context.get("workspace") or ""),
                Path(str(context.get("output_dir") or "")),
                "forge_experiments",
            )
        except Exception as exc:  # noqa: BLE001
            preserve_error = exc
        try:
            _restore_inplace(context.get("restore"))
            restore = context.get("restore") or {}
            source_file = str(restore.get("source_file") or "")
            backup = restore.get("backup")
            if source_file and isinstance(backup, bytes) and Path(source_file).read_bytes() != backup:
                raise RuntimeError(f"collective workspace restore did not recover {source_file}")
            repo = str(context.get("kernel_repo") or "")
            _restore_config(
                repo,
                dict(context.get("config_snapshot") or {}),
            )
            _verify_restored_repo(repo, restore)
            _restore_journal_path(repo).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            if preserve_error is not None:
                raise RuntimeError(
                    f"workspace restore failed after campaign preservation failed: {preserve_error!r}"
                ) from exc
            raise
        if preserve_error is not None:
            raise preserve_error
        return
    try:
        _remove_verified_worktree(
            str(context.get("kernel_repo") or ""),
            str(context.get("source_file") or ""),
            str(context.get("workspace") or ""),
            str(context.get("branch") or ""),
        )
    finally:
        _restore_config(
            str(context.get("kernel_repo") or ""),
            dict(context.get("config_snapshot") or {}),
        )


def _load_forge_result(output_dir: str) -> dict[str, Any]:
    """Load and validate forge-loop's result object when present."""
    path = Path(output_dir or ".") / "forge_result.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid forge result: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"forge result must be a JSON object: {path}")
    return payload


def _validated_published_best(context: dict[str, Any]) -> dict[str, Any]:
    """Read KernelForge's authoritative crash-safe best-result sidecar."""
    workspace = str(context.get("workspace") or "")
    base_commit = str(context.get("base_commit") or "")
    published = _read_forge_best_result(workspace)
    validated = _validated_forge_best_result(
        published,
        workspace=workspace,
        base_commit=base_commit,
    )
    return dict(validated or {})


def _best_commit(payload: dict[str, Any]) -> str:
    """Return KernelForge's canonical best commit."""
    return str(payload.get("best_commit") or "").strip()


def _export_collective_result(
    context: dict[str, Any],
    output_dir: str,
    payload: dict[str, Any],
) -> tuple[str, list[str], str]:
    """Export a validated best commit before the temporary branch is restored."""
    improved = payload.get("improved") is True
    best_commit = _best_commit(payload)
    if not improved or not best_commit:
        return "", [], best_commit
    workspace = str(context.get("workspace") or "")
    base_commit = str(context.get("base_commit") or "")
    ancestry = _git(
        workspace,
        "merge-base",
        "--is-ancestor",
        base_commit,
        best_commit,
    )
    if not base_commit or ancestry.returncode != 0:
        raise RuntimeError("validated collective best is not descended from this campaign baseline")
    _, changed = _export_best_artifacts(
        workspace,
        base_commit,
        str(context.get("prepared_source") or ""),
        str(context.get("source_file") or ""),
        Path(output_dir),
        best_commit=best_commit,
    )
    patch = Path(output_dir) / "optimized_versions" / "forge.patch"
    if not patch.is_file() or not patch.read_text(encoding="utf-8", errors="replace").strip():
        raise RuntimeError("validated collective best produced no exportable patch")
    return str(patch), list(changed), best_commit


def _base_result(output_dir: str) -> dict[str, Any]:
    """Return the failed result envelope used by every wrapper error path."""
    return {
        "status": "failed",
        "engine": "forge_collective",
        "micro_decision": "failed",
        "decision": "REVERT",
        "kept": False,
        "kernel_speedup": None,
        "env_flags": {},
        "artifact_files": [],
        "patch": None,
        "requires_e2e_validation": False,
        "workspace": str(output_dir or ""),
    }


def _validated_exported_patch(output_dir: str, patch_path: str) -> str:
    """Return an exported Forge patch confined to the output directory."""
    if not patch_path:
        return ""
    expected_root = (Path(output_dir) / "optimized_versions").resolve()
    patch = Path(patch_path).resolve()
    try:
        patch.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"exported patch is outside the output directory: {patch}") from exc
    if not patch.is_file():
        raise ValueError(f"exported patch does not exist: {patch}")
    try:
        text = patch.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read exported patch: {patch}") from exc
    if not text.startswith("diff --git "):
        raise ValueError(f"exported patch has invalid content: {patch}")
    return str(patch)


def _normalize_result(
    output_dir: str,
    rc: int,
    rig: dict[str, str],
    *,
    source_file: str = "",
    kernel_repo: str = "",
    patch_path: str = "",
    changed_files: list[str] | None = None,
    best_commit: str = "",
    result_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map forge-loop's real result schema onto Hyperloom's result contract."""
    result = _base_result(output_dir)
    result["collective_op"] = rig.get("collective_op", "")
    result["world_size"] = rig.get("world_size", "")
    result["driver"] = rig.get("driver", "")
    # Empty marks a campaign that authored its driver from the trace alone.
    result["invocation_spec"] = rig.get("invocation_spec", "")

    path = Path(output_dir or ".") / "forge_result.json"
    if result_payload is None and not path.is_file():
        result["error_class"] = "missing_forge_result"
        result["error"] = f"no forge_result.json at {path} (forge-loop rc={rc})"
        return result
    try:
        if result_payload is not None and not isinstance(result_payload, dict):
            raise ValueError("Collective result payload must be a mapping")
        payload = dict(result_payload) if result_payload is not None else _load_forge_result(output_dir)
    except ValueError as exc:
        result["error_class"] = "invalid_forge_result"
        result["error"] = str(exc)
        return result
    if not payload:
        result["error_class"] = "empty_forge_result"
        result["error"] = "forge_result.json is an empty JSON object"
        return result

    bandwidth = payload.get("case_bandwidth")
    if isinstance(bandwidth, dict):
        result["bandwidth"] = bandwidth

    forge_error = payload.get("error")
    if forge_error is not None and (not isinstance(forge_error, str) or not forge_error.strip()):
        result["error_class"] = "invalid_forge_result"
        result["error"] = "forge result has invalid error"
        return result
    improved = payload.get("improved")
    if (improved is not None and not isinstance(improved, bool)) or (forge_error is None and improved is None):
        result["error_class"] = "invalid_forge_result"
        result["error"] = "forge result requires boolean improved when no error"
        return result
    speedup_raw = payload.get("mean_case_speedup")
    if speedup_raw is not None and (isinstance(speedup_raw, bool) or not isinstance(speedup_raw, (int, float))):
        result["error_class"] = "invalid_forge_result"
        result["error"] = "forge result has invalid mean_case_speedup"
        return result
    speedup = None if speedup_raw is None else float(speedup_raw)
    if speedup is not None and (not math.isfinite(speedup) or speedup <= 0.0):
        result["error_class"] = "invalid_forge_result"
        result["error"] = "forge result has invalid mean_case_speedup"
        return result
    requested_keep = improved is True
    if requested_keep and (speedup is None or speedup <= 1.0):
        result["error_class"] = "invalid_forge_result"
        result["error"] = "improved forge result requires mean_case_speedup greater than one"
        return result
    if improved is False and speedup is not None and speedup > 1.0:
        result["error_class"] = "invalid_forge_result"
        result["error"] = "forge result has inconsistent improvement fields"
        return result
    iteration_key = "iteration" if payload.get("source") == "best_result.json" else "iteration_count"
    iteration_raw = payload.get(iteration_key)
    if iteration_raw is not None and (isinstance(iteration_raw, bool) or not isinstance(iteration_raw, int)):
        result["error_class"] = "invalid_forge_result"
        result["error"] = f"forge result has invalid {iteration_key}"
        return result
    iterations = iteration_raw
    if iterations is not None and iterations < 0:
        result["error_class"] = "invalid_forge_result"
        result["error"] = f"forge result has invalid {iteration_key}"
        return result
    changed = list(changed_files or [])
    try:
        patch = _validated_exported_patch(output_dir, patch_path)
    except ValueError as exc:
        result["error_class"] = "invalid_exported_patch"
        result["error"] = str(exc)
        return result
    kept = requested_keep and bool(patch) and bool(best_commit)
    failed = bool(forge_error) or (rc != 0 and not kept)

    result.update(
        {
            "status": ("ok" if kept else ("failed" if failed or requested_keep else "complete")),
            "micro_decision": ("candidate" if kept else ("failed" if failed or requested_keep else "no_improvement")),
            "decision": "KEEP" if kept else "REVERT",
            "kept": kept,
            "kernel_speedup": speedup,
            "artifact_files": list(changed) if kept else [],
            "patch": patch or None,
            "source_file": str(source_file),
            "kernel_repo": str(kernel_repo),
            "iterations": iterations,
            "experiment_id": payload.get("experiment_id"),
            "best_commit": best_commit if kept else "",
            "salvaged": bool(kept and rc != 0),
            # Kernel parity only; integrate confirms the real end-to-end gain.
            "requires_e2e_validation": kept,
        }
    )
    if forge_error and not kept:
        result["error_class"] = str(forge_error)
        result["error"] = str(payload.get("detail") or forge_error)
    elif rc != 0 and not kept:
        result["error_class"] = "forge_loop_failed"
        result["error"] = f"forge-loop exited with rc={rc} without a validated best"
    elif requested_keep and not kept:
        result["error_class"] = "unverified_collective_improvement"
        result["error"] = "forge-loop improvement lacks a validated commit and exported patch"
    return result


def _timeout_result(output_dir: str, timeout_sec: int, exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    """Build the structured result for a campaign timeout."""
    result = _base_result(output_dir)
    cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or []))
    result["error_class"] = "subprocess_timeout"
    result["error"] = f"TimeoutExpired after {timeout_sec}s: {cmd_repr[:1500]}"
    return result


def _relay(stdout: Any, stderr: Any) -> None:
    """Relay captured Forge output to the wrapper streams."""
    if stdout:
        sys.stdout.write(stdout if isinstance(stdout, str) else stdout.decode("utf-8", "replace"))
    if stderr:
        sys.stderr.write(stderr if isinstance(stderr, str) else stderr.decode("utf-8", "replace"))


def _emit(result: dict[str, Any], output_dir: str) -> None:
    """Persist and print the wrapper result contract."""
    if output_dir:
        try:
            (Path(output_dir) / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"warning: failed to persist result.json: {exc}", file=sys.stderr, flush=True)
    print(f"\n{RESULT_BEGIN}\n{json.dumps(result, sort_keys=True)}\n{RESULT_END}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run one collective Forge campaign with export-before-restore safety."""
    parser = argparse.ArgumentParser(description="Hyperloom wrapper for forge-loop (collective)")
    parser.add_argument("--input-json", required=True)
    output_dir = ""
    context: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        args = parser.parse_args(list(argv or sys.argv[1:]))
        payload = _load_input_json(args.input_json)
        output_dir = str(payload.get("output_dir") or "")
        if not output_dir:
            raise ValueError("output_dir is required")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        context = _prepare_collective_workspace(payload, Path(output_dir))
        if context is None:
            result = _base_result(output_dir)
            result.update(
                {
                    "status": "skipped",
                    "error_class": "collective_workspace_unavailable",
                    "error": "collective repository is unusable or locked by another Forge run",
                }
            )
            return_code = 0
        else:
            prepared_payload = dict(payload)
            candidate_raw = payload.get("candidate")
            if not isinstance(candidate_raw, dict):
                raise ValueError("candidate must be a mapping")
            candidate = dict(candidate_raw)
            candidate["source_file"] = context["prepared_source"]
            candidate["kernel_repo"] = context["workspace"]
            prepared_payload.update(
                {
                    "candidate": candidate,
                    "source_file": context["prepared_source"],
                    "kernel_repo": context["workspace"],
                    "git_branch": context["branch"],
                }
            )
            rig = generate_collective_driver(
                candidate,
                output_dir,
                tp=payload.get("tp"),
            )
            prepared_payload["invocation_spec_file"] = _write_invocation_evidence(
                candidate,
                Path(output_dir),
            )
            rig["invocation_spec"] = prepared_payload["invocation_spec_file"]
            timeout_sec = _campaign_timeout_sec(
                prepared_payload,
                time.monotonic() - started,
            )
            cmd = _build_cmd(
                prepared_payload,
                rig,
                Path(output_dir),
                deadline_unix=(
                    int(time.time())
                    + max(
                        1,
                        timeout_sec - FORGE_SHUTDOWN_GRACE_SEC,
                    )
                ),
            )
            _inject_author_gateway_env()
            timed_out: subprocess.TimeoutExpired | None = None
            try:
                proc = _run_with_tree_timeout(cmd, timeout_sec)
                return_code = int(proc.returncode)
                stdout, stderr = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = exc
                return_code = 124
                stdout = getattr(exc, "stdout", None)
                stderr = getattr(exc, "stderr", None)

            _relay(stdout, stderr)
            _persist_logs(output_dir, stdout, stderr)
            published_best = _validated_published_best(context)
            try:
                forge_payload = _load_forge_result(output_dir)
            except ValueError:
                if not published_best and timed_out is None:
                    raise
                forge_payload = {}
            if published_best:
                # A published best supersedes a wrapper failure but does not
                # erase it: the KEEP still came out of an unclean exit.
                superseded = {
                    key: forge_payload[key] for key in ("error", "detail") if forge_payload.get(key) not in (None, "")
                }
                forge_payload = {**forge_payload, **published_best}
                forge_payload.pop("error", None)
                forge_payload.pop("detail", None)
                if superseded:
                    forge_payload["superseded_error"] = superseded
            patch_path = ""
            changed_files: list[str] = []
            best_commit = ""
            export_error: Exception | None = None
            try:
                patch_path, changed_files, best_commit = _export_collective_result(
                    context,
                    output_dir,
                    forge_payload,
                )
            except Exception as exc:  # noqa: BLE001
                export_error = exc

            if timed_out is not None and not patch_path and export_error is None:
                result = _timeout_result(output_dir, timeout_sec, timed_out)
                result.update(
                    {
                        "collective_op": rig.get("collective_op", ""),
                        "world_size": rig.get("world_size", ""),
                        "driver": rig.get("driver", ""),
                    }
                )
                if forge_payload:
                    result["partial_forge_result"] = forge_payload
            else:
                result = _normalize_result(
                    output_dir,
                    return_code,
                    rig,
                    source_file=str(context["source_file"]),
                    kernel_repo=str(context["kernel_repo"]),
                    patch_path=patch_path,
                    changed_files=changed_files,
                    best_commit=best_commit,
                    result_payload=forge_payload or None,
                )
                if export_error is not None:
                    result.update(
                        {
                            "status": "failed",
                            "decision": "REVERT",
                            "kept": False,
                            "requires_e2e_validation": False,
                            "error_class": "collective_export_failed",
                            "error": repr(export_error),
                        }
                    )
    except Exception as exc:  # noqa: BLE001 - structured wrapper failure
        result = _base_result(output_dir)
        result.update(
            {
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        )
        return_code = 2
    finally:
        if context is not None:
            try:
                _restore_collective_workspace(context)
            except Exception as exc:  # noqa: BLE001
                prior = dict(result or {})
                result = _base_result(output_dir)
                result.update(
                    {
                        "error_class": "collective_workspace_restore_failed",
                        "error": repr(exc),
                        "pre_restore_result": prior,
                    }
                )
                return_code = 2

    if result is None:
        result = _base_result(output_dir)
        result["error"] = "collective wrapper completed without a result"
        return_code = 2
    _emit(result, output_dir)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
