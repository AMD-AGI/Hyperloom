# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-baselining a patched tree, and the one retry a JIT registry miss or a stalled baton earns."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator, NamedTuple

from hyperloom.common.coerce import to_str_list

from . import patch_jit_cache
from ..actions.executors._kernel_agent_tool import HandlerResult
from ._integrate_outcomes import _FRAMEWORK_APPLYBACK_ARTIFACT_KIND, _FailureVerdict, _integrate_failure
from ._integrate_payload import _is_paired_gemm_measurement
from .kernel_decisions import _honest_flag

log = logging.getLogger(__name__)


def _vram_guarded_server_args(extra_args: str) -> str:
    """Optionally cap ``--gpu-memory-utilization`` for the integrate re-baseline.

    When ``HL_INTEGRATE_VRAM_GUARD`` is on and the caller has not already pinned
    ``--gpu-memory-utilization``, append a conservative cap
    (``HL_INTEGRATE_VRAM_UTIL_CAP``, default 0.90) so a re-baseline server cannot
    OOM. A strict no-op when the flag is off or a util is already specified.
    """
    if not _honest_flag("HL_INTEGRATE_VRAM_GUARD"):
        return extra_args
    # ``--gpu-memory-utilization`` is vLLM-only; apply the cap only for vLLM.
    framework = (os.environ.get("FRAMEWORK") or "").strip().lower()
    if framework != "vllm":
        return extra_args
    if "gpu-memory-utilization" in (extra_args or ""):
        return extra_args
    try:
        cap = float(os.environ.get("HL_INTEGRATE_VRAM_UTIL_CAP", "0.90") or 0.90)
    except (TypeError, ValueError):
        cap = 0.90
    cap = min(max(cap, 0.1), 0.99)
    addition = f"--gpu-memory-utilization {cap:g}"
    return f"{extra_args} {addition}".strip() if extra_args else addition


def _sweep_integrate_aiter_locks(*, reason: str) -> dict[str, Any]:
    """Best-effort orphaned-lock sweep immediately before an integrate boot."""
    from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

    try:
        stats = sweep_stale_aiter_locks_if_dead()
    except (OSError, RuntimeError) as exc:
        log.warning("integrate_handler: aiter lock sweep failed before %s: %r", reason, exc)
        return {"errors": 1, "exception": repr(exc)}
    if stats.get("skipped_live"):
        log.info(
            "integrate_handler: aiter lock sweep skipped before %s; a compiler is alive",
            reason,
        )
    elif stats.get("deleted"):
        log.warning(
            "integrate_handler: reaped %d orphaned aiter lock(s) across %s before %s",
            stats.get("deleted"),
            stats.get("dirs") or [stats.get("dir")],
            reason,
        )
    return stats


_INTEGRATE_LOG_NAMES = ("server.log", "benchmark_stderr.log", "benchmark_stdout.log")


def _workspace_log_sizes(workspace: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name in _INTEGRATE_LOG_NAMES:
        path = workspace / name
        try:
            sizes[name] = path.stat().st_size if path.is_file() else 0
        except OSError:
            sizes[name] = 0
    return sizes


def _compiled_registry_error_text(
    workspace: Path,
    *,
    after_sizes: dict[str, int] | None = None,
) -> str:
    """Return the integrate log excerpt that shows a compiled-registry miss.

    The excerpt names the kernel aiter could not find, which is the only thing that
    identifies the module at fault when the round's env does not.
    """
    from ..actions.executors._aiter_jit import is_aiter_jit_registry_mismatch

    for name in _INTEGRATE_LOG_NAMES:
        path = workspace / name
        if not path.is_file():
            continue
        start = 0 if after_sizes is None else max(0, int(after_sizes.get(name, 0)))
        try:
            with path.open("rb") as handle:
                if after_sizes is None:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 65536))
                else:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if start >= size:
                        handle.seek(max(0, size - 65536))
                    else:
                        handle.seek(start)
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        if is_aiter_jit_registry_mismatch(text):
            return text
    return ""


def _workspace_has_compiled_registry_error(
    workspace: Path,
    *,
    after_sizes: dict[str, int] | None = None,
) -> bool:
    """True when an integrate workspace log shows a compiled-registry miss."""
    return bool(_compiled_registry_error_text(workspace, after_sizes=after_sizes))


def _integrate_extra_envs(ctx: Any) -> dict[str, str] | None:
    task = getattr(ctx, "task", None)
    params = getattr(task, "params", None)
    if isinstance(params, dict):
        envs = params.get("extra_envs")
        if isinstance(envs, dict):
            return envs
    return None


def _lock_cleanup_allows_retry(cleanup: dict[str, Any]) -> bool:
    if cleanup.get("skipped_live") or cleanup.get("errors"):
        return False
    return bool(cleanup.get("deleted")) or (cleanup.get("scanned", 0) == 0 and not cleanup.get("skipped_fresh"))


async def _run_integrate_rebaseline_with_lock_retry(
    executor: Any,
    ctx: Any,
    *,
    workspace: Path,
    reason: str,
) -> dict[str, Any]:
    """Run one integrate baseline and retry once after a JIT registry miss or baton stall."""
    from ..actions.executors._aiter_jit import (
        drop_serving_so_for_envs,
        find_aiter_baton_wait,
        result_is_aiter_jit_registry_mismatch,
    )

    prelaunch_sweep = _sweep_integrate_aiter_locks(reason=reason)
    first_started_unix = time.time()
    result = await executor(ctx)
    if not isinstance(result, dict) or result.get("status") == "succeeded":
        return result

    if result_is_aiter_jit_registry_mismatch(result) or _workspace_has_compiled_registry_error(workspace):
        envs = _integrate_extra_envs(ctx)
        log_sizes = _workspace_log_sizes(workspace)
        # The env does not always reach the module at fault: a round that pins one
        # table leaves every other AITER_CONFIG_* unset, and each of those resolves to
        # the shipped default plus its matching model_configs overlays, and
        # AITER_CONFIG_FMOE maps to no serving module at all. Read the module out of
        # the kernel the error named, or the drop unlinks nothing and the retry
        # repeats the failure.
        from ..actions.executors._aiter_jit import registry_mismatch_modules

        named_modules = registry_mismatch_modules(
            str(result.get("error") or ""),
            _compiled_registry_error_text(workspace),
        )
        cleanup = drop_serving_so_for_envs(
            envs,
            backup_dir=workspace / "aiter_jit_backup",
            also_modules=named_modules,
        )
        log.warning(
            "integrate_handler: classified %s as aiter_jit_registry_mismatch; "
            "retrying once after so drop (env modules + %s from the error)",
            reason,
            list(named_modules) or "none",
        )
        retry_result = await executor(ctx)
        if not isinstance(retry_result, dict):
            return retry_result
        retry_result["aiter_jit_registry_mismatch_retry"] = {
            "cleanup": cleanup,
            "retry_attempted": True,
            "retry_succeeded": retry_result.get("status") == "succeeded",
        }
        if result_is_aiter_jit_registry_mismatch(retry_result) or (
            retry_result.get("status") != "succeeded"
            and _workspace_has_compiled_registry_error(workspace, after_sizes=log_sizes)
        ):
            retry_result["error_class"] = "aiter_jit_registry_mismatch"
        return retry_result

    evidence = find_aiter_baton_wait(
        workspace,
        since_unix=first_started_unix - 1.0,
    )
    if evidence is None:
        return result

    cleanup = _sweep_integrate_aiter_locks(reason=f"{reason} stale-lock retry")
    result["error_class"] = "stale_jit_lock"
    result["stale_jit_lock"] = {
        "evidence": evidence,
        "prelaunch_sweep": prelaunch_sweep,
        "post_failure_sweep": cleanup,
        "retry_attempted": False,
    }

    if not _lock_cleanup_allows_retry(cleanup):
        return result

    log.warning(
        "integrate_handler: classified %s as stale_jit_lock; retrying once after cleanup",
        reason,
    )
    retry_started_unix = time.time()
    retry_result = await executor(ctx)
    if not isinstance(retry_result, dict):
        return retry_result
    retry_evidence = (
        find_aiter_baton_wait(
            workspace,
            since_unix=retry_started_unix - 1.0,
        )
        if retry_result.get("status") != "succeeded"
        else None
    )
    retry_result["stale_jit_lock_retry"] = {
        "evidence": evidence,
        "cleanup": cleanup,
        "retry_attempted": True,
        "retry_succeeded": retry_result.get("status") == "succeeded",
    }
    if retry_evidence is not None:
        retry_result["error_class"] = "stale_jit_lock"
        retry_result["stale_jit_lock_retry"]["retry_evidence"] = retry_evidence
    return retry_result


class _IntegrateRebaseline(NamedTuple):
    workspace: Path
    server_args: str
    executor: Any
    ctx: Any


def _integrate_rebaseline(
    payload: dict,
    state: Any,
    *,
    session_dir: Path,
    performance_policy: dict[str, float],
) -> _IntegrateRebaseline:
    from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id
    from hyperloom.inference_optimizer.session.session_paths import fs_safe_id, unique_runs_dir
    from ..actions.executors.baseline import BaselineExecutor
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    # VRAM barrier (HL_HONEST_E2E umbrella, default ON; opt out with
    # HL_HONEST_E2E=0 or HL_INTEGRATE_VRAM_GUARD=0): cap re-baseline util on
    # vLLM so the integrate server cannot OOM on a tighter node.
    extra_args = _vram_guarded_server_args(str(payload.get("extra_server_args") or "").strip())
    # A fusion sibling's kernel_id is its operator name (``llm:<recipe>``), which
    # only reaches this handler now that fusion lands through the generic queue.
    # It is a legal id but not a legal directory name everywhere -- fold it.
    task_id = f"integrate-{fs_safe_id(payload.get('kernel_id'))}"
    workspace = unique_runs_dir(session_dir, "integrate", task_id)
    keep_policy = (
        {} if _is_paired_gemm_measurement(payload) else {"post_measure_accuracy_keep_policy": performance_policy}
    )
    task = Task(
        task_id=task_id,
        kind="baseline",
        state="running",
        params={
            "config_path": payload.get("config_path"),
            "output_dir": str(workspace),
            "extra_server_args": extra_args,
            "extra_envs": dict(payload.get("extra_envs") or {}),
            "remove_args": to_str_list(payload.get("remove_args")),
            "unset_envs": to_str_list(payload.get("unset_envs")),
            "args_mode": str(payload.get("args_mode") or "append"),
            # The only artifact that patches FlyDSL sources, so the only run that
            # needs the JIT cache key widened.
            "flydsl_source_dirs": (str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND),
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": performance_policy["base_tput"]
            * (1.0 + performance_policy["keep_threshold_pct"] / 100.0),
            **keep_policy,
            # Synthetic kind="baseline": candidate A/B validation against the
            # already-anchored reference. It runs eval for the kernel accuracy
            # gate but never establishes a replacement quality reference.
            "quality_ref_exempt": True,
            # A sub-step of the KERNEL phase's own event, not a dispatched
            # measurement, so it records into that event rather than leaving a
            # baseline event of its own.
            INLINE_EVENT_PARAM: kernel_event_id(int(getattr(state, "macro_cycle", 0) or 0)),
        },
        idempotency_key=f"{task_id}-rebaseline",
    )
    return _IntegrateRebaseline(
        workspace=workspace,
        server_args=extra_args,
        executor=BaselineExecutor(session_dir=session_dir, shared_state=state),
        ctx=RunnerContext(task=task, lease=None),
    )


@contextlib.contextmanager
def _forced_cpp_itfs_rebuild(apply_result: dict) -> Iterator[None]:
    if not (apply_result.get("cpp_itfs_cache_backup") or {}).get("is_cpp_itfs"):
        yield
        return
    previous = os.environ.get("AITER_REBUILD")
    os.environ["AITER_REBUILD"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AITER_REBUILD", None)
        else:
            os.environ["AITER_REBUILD"] = previous


async def _multi_node_restart_failure(
    rebaseline: _IntegrateRebaseline,
    payload: dict,
    apply_result: dict,
) -> HandlerResult | None:
    from ..actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return None
    # The sweep runs even when the regular JIT cache is warm: cpp_itfs attention
    # uses the separate AITER_ROOT_DIR/build tree and can carry a stale baton
    # from a timed-out Forge driver.
    _sweep_integrate_aiter_locks(reason="integrate server startup")
    from ..actions.executors._multi_node_server_lifecycle import (
        RoundServer,
        ServerRestartFailed,
        restart_server_for_round,
    )

    try:
        await restart_server_for_round(
            RoundServer.from_env(
                rebaseline.server_args,
                (str(payload.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None),
            ),
            force_full_restart=True,
        )
    except ServerRestartFailed as exc:
        return _integrate_failure(
            payload, apply_result, _FailureVerdict("mn_server_restart_failed_post_patch", str(exc), "REVERT")
        )
    ctx = rebaseline.ctx
    ctx.extra = {**(getattr(ctx, "extra", None) or {}), "mn_round_restarted": True}
    return None


def _cpp_itfs_rebuild_check(apply_result: dict) -> HandlerResult:
    from ..actions.executors._multi_node_env import is_multi_node

    cpp_itfs_backup = apply_result.get("cpp_itfs_cache_backup") or {}
    if not cpp_itfs_backup.get("is_cpp_itfs") or is_multi_node():
        return {"verified": True, "status": "skipped"}
    return patch_jit_cache.verify_cpp_itfs_rebuilt(cpp_itfs_backup)
