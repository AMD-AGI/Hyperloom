# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node-only: per-round sglang/vllm restart helper."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple

from hyperloom.common.env import env_flag
from hyperloom.inference_optimizer.multi_node._internal.env_safety import filter_forward_env
from hyperloom.inference_optimizer.multi_node._internal.server_args_safety import (
    ServerArgsRejected,
    validate_server_args,
)
from ._multi_node_env import _read_state, is_multi_node
from ._multi_node_server_readiness import ServerRestartFailed, wait_for_restarted_server
from ._multi_node_worker_logs import collect_worker_server_logs

# Scoped env flag (set by the roofline compute-bound re-profile) that tells this restart to strip DP-attention /
# dp-size from prefill+decode+shared server args, so the profiled server runs single-rank full-batch (compute-bound).
_COMPUTE_BOUND_PROFILE_ENV = "HYPERLOOM_MN_PROFILE_COMPUTE_BOUND"

# Control variables the multi-node launcher reads out of this process' environment for one restart.
_PROFILE_TRACE_DIR_ENV = "HYPERLOOM_MN_PROFILE_TRACE_DIR"
_EXTRA_FWD_ENV = "HYPERLOOM_MN_EXTRA_FWD_ENV"
_UNSET_FWD_ENV = "HYPERLOOM_MN_UNSET_FWD_ENV"
_RESUME_RUNNING_ENV = "MULTI_NODE_RESTART_RESUME_RUNNING"

# Gate for the reclaim-and-retry path in ``restart_server_for_round``: when a restart fails, best-effort remote
# ``kill-inference`` (Infera SSH fan-out / RayJob Dashboard kill job) reclaims VRAM pinned by a crashed prior server,
# then exactly one forced fresh restart is retried.
_MN_RESTART_RECLAIM_RETRY_ENV = "HYPERLOOM_MN_RESTART_RECLAIM_RETRY"


def _apply_env(values: Mapping[str, str | None]) -> None:
    """Set each variable, removing the ones whose value is ``None``."""
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@contextmanager
def _scoped_env(values: Mapping[str, str | None]) -> Iterator[None]:
    """Apply each variable for the block, restoring what was there on exit."""
    saved: dict[str, str | None] = {name: os.environ.get(name) for name in values}
    try:
        _apply_env(values)
        yield
    finally:
        _apply_env(saved)


def _strip_dp_parallel_flags(extra_args: str) -> str:
    """Remove DP-attention / dp-size flags from a server-args string."""
    try:
        toks = shlex.split(extra_args or "")
    except ValueError:
        return extra_args or ""
    out: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in ("--enable-dp-attention", "--enable-dp-lm-head"):
            i += 1
            continue
        if tok == "--dp-size":
            i += 2  # skip flag and its value
            continue
        if tok.startswith("--dp-size="):
            i += 1
            continue
        out.append(tok)
        i += 1
    return " ".join(shlex.quote(x) for x in out)


log = logging.getLogger(__name__)


# Default /health poll timeout; override per-run via HYPERLOOM_MN_HEALTH_WAIT_S.
DEFAULT_HEALTH_TIMEOUT_S = 900  # 15 min.
_POLL_INTERVAL_S = 6

# Magpie's sglang_mi*x.sh DEFAULT_ARGS, re-applied in multi-node so tput stays comparable to single-node.
# --mem-fraction-static is 0.75 (vs 0.8) because cross-node RDMA buffers eat headroom.
_SGLANG_DEFAULT_TOKENS: tuple[tuple[str, str], ...] = (
    ("--mem-fraction-static", "--mem-fraction-static=0.75"),
    ("--disable-radix-cache", "--disable-radix-cache"),
)


def _merge_sglang_defaults(extra_args: str) -> str:
    """Append Magpie's DEFAULT_ARGS that the user did not already set."""
    user = (extra_args or "").strip()
    parts = [user] if user else []
    for flag_name, default_token in _SGLANG_DEFAULT_TOKENS:
        if flag_name in user:
            continue
        parts.append(default_token)
    return " ".join(p for p in parts if p)


class _PdGroup(NamedTuple):
    """One PD role's launch knobs: how many nodes it gets, and how each of them is configured."""

    nodes: int
    tp: int
    ep: int
    extra_args: str


def _state_or_env_int(state: Mapping[str, Any], state_key: str, env_key: str) -> int:
    """Read an int knob from state.json, falling back to its environment variable, else 0."""
    for raw in (state.get(state_key), os.environ.get(env_key, "")):
        if not raw:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


def _state_or_env_text(state: Mapping[str, Any], state_key: str, env_key: str) -> str:
    """Read a text knob from state.json, falling back to its environment variable."""
    return str(state.get(state_key) or os.environ.get(env_key, "") or "").strip()


def _recorded_pod_count(state: Mapping[str, Any], role: str) -> int:
    """How many pods state.json records for a PD role, from whichever of its lists is populated."""
    for key in (role + "_pod_ips", role + "_pods"):
        pods = state.get(key)
        if pods:
            return len(pods)
    return 0


def _resolve_pd_group(state: Mapping[str, Any], role: str, *, tp_int: int) -> _PdGroup:
    """Resolve one PD role's node / TP / EP / extra-args knobs; 0 and "" fall back to the shared --ep / --extra-args."""
    state_prefix = "last_restart_pd_" + role
    env_prefix = "PD_" + role.upper()
    nodes = _state_or_env_int(state, state_prefix + "_nodes", env_prefix + "_NODES")
    # The restart path launches with ``nodes or len(pods)`` but persists the raw arg (often 0), so a resume that also
    # lost the ``$PD_*_NODES`` env would leave nodes at 0 and wrongly fail the disaggregated gate (e.g. auto-roofline
    # after resume).
    return _PdGroup(
        nodes=nodes if nodes > 0 else _recorded_pod_count(state, role),
        tp=_state_or_env_int(state, state_prefix + "_tp", env_prefix + "_TP") or tp_int,
        ep=_state_or_env_int(state, state_prefix + "_ep", env_prefix + "_EP"),
        extra_args=_state_or_env_text(state, state_prefix + "_extra_args", env_prefix + "_EXTRA_ARGS"),
    )


def _validate_pd_layout(prefill: _PdGroup, decode: _PdGroup, *, state_nodes: int, ep_int: int) -> None:
    """Reject a resolved disaggregated layout the launcher could not place."""
    if prefill.nodes <= 0 or decode.nodes <= 0:
        raise ServerRestartFailed(
            f"pd_mode=disaggregated requires pd_prefill_nodes>0 and pd_decode_nodes>0; "
            f"got pn={prefill.nodes} dn={decode.nodes}"
        )
    if state_nodes > 0 and prefill.nodes + decode.nodes != state_nodes:
        raise ServerRestartFailed(
            f"pd_prefill_nodes ({prefill.nodes}) + pd_decode_nodes ({decode.nodes}) "
            f"must equal total nodes ({state_nodes})"
        )
    if prefill.tp <= 0 or decode.tp <= 0:
        raise ServerRestartFailed(f"pd_prefill_tp ({prefill.tp}) and pd_decode_tp ({decode.tp}) must be positive")
    min_grp_tp = min(prefill.tp, decode.tp)
    if ep_int > 1 and ep_int > min_grp_tp:
        raise ServerRestartFailed(
            f"ep={ep_int} > min(pd_prefill_tp={prefill.tp}, "
            f"pd_decode_tp={decode.tp})={min_grp_tp}; "
            "lower --ep or raise the smaller per-group TP."
        )


def _resolve_pd_args(*, tp_int: int, ep_int: int) -> dict:
    """Resolve PD knobs with state.json + env fallback."""
    state = _read_state()
    mode = (state.get("last_restart_pd_mode") or os.environ.get("PD_MODE", "") or "aggregated").strip().lower()
    if mode not in ("aggregated", "disaggregated"):
        raise ServerRestartFailed(f"unsupported pd_mode {mode!r}; expected 'aggregated' or 'disaggregated'")

    out: dict = {"pd_mode": mode}
    if mode == "aggregated":
        return out

    # PD disaggregation requires >=2 nodes; defend against a mangled state.json.
    state_nodes = int(state.get("nodes") or 0)
    if state_nodes < 2:
        raise ServerRestartFailed(
            f"pd_mode=disaggregated requires nodes>=2 but state.json "
            f"reports nodes={state_nodes}. Re-provision the RayJob "
            "with `--nodes >=2` or drop --pd-mode."
        )

    prefill = _resolve_pd_group(state, "prefill", tp_int=tp_int)
    decode = _resolve_pd_group(state, "decode", tp_int=tp_int)
    _validate_pd_layout(prefill, decode, state_nodes=state_nodes, ep_int=ep_int)
    out.update(
        {
            "pd_prefill_nodes": prefill.nodes,
            "pd_decode_nodes": decode.nodes,
            "pd_prefill_tp": prefill.tp,
            "pd_decode_tp": decode.tp,
            "pd_transfer_backend": _state_or_env_text(state, "last_restart_pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            "pd_ib_device": _state_or_env_text(state, "last_restart_pd_ib_device", "PD_IB_DEVICE"),
            "pd_prefill_ep": prefill.ep,
            "pd_decode_ep": decode.ep,
            "pd_prefill_extra_args": prefill.extra_args,
            "pd_decode_extra_args": decode.extra_args,
        }
    )
    return out


def _resolve_parallel_width(explicit: int | None, state_value: object, env_key: str, on_error: int) -> int:
    """A TP/EP width from the explicit argument, state.json or the environment, falling back to ``on_error``."""
    try:
        return int(explicit if explicit is not None else (state_value or os.environ.get(env_key, "") or on_error))
    except (TypeError, ValueError):
        return on_error


def _resolve_round_args(
    framework: str | None,
    model_path: str | None,
    tp: int | None,
    ep: int | None = None,
) -> tuple[str, str, int, int]:
    """Resolve (framework, model, tp, ep) for the restart, with state fallback."""
    state = _read_state()
    fw = (
        (framework or state.get("last_restart_framework") or os.environ.get("FRAMEWORK", "sglang") or "sglang")
        .strip()
        .lower()
    )
    mdl = (model_path or state.get("last_restart_model") or os.environ.get("MODEL_PATH", "") or "").strip()
    tp_int = _resolve_parallel_width(tp, state.get("last_restart_tp"), "TP", 0)
    ep_int = _resolve_parallel_width(ep, state.get("last_restart_ep"), "EP", 1)
    if ep_int < 1:
        ep_int = 1
    if not mdl or tp_int <= 0:
        raise ServerRestartFailed(
            "cannot restart multi-node server: missing model/tp "
            f"(framework={fw!r} model={mdl!r} tp={tp_int}). "
            "Pass model_path/tp explicitly or run "
            "`multi_node restart-server` once so state.json is populated."
        )
    if fw not in ("sglang", "vllm"):
        raise ServerRestartFailed(f"unsupported framework {fw!r}; expected 'sglang' or 'vllm'")
    if ep_int > tp_int:
        raise ServerRestartFailed(
            f"ep={ep_int} > tp={tp_int} is not supported by sglang/vllm "
            "(cannot place more expert shards than ranks). Lower --ep or "
            "raise --tp."
        )
    return fw, mdl, tp_int, ep_int


@dataclass(frozen=True)
class _ResolvedRound:
    """What one multi-node restart launches: framework, model, parallelism, PD layout and server args."""

    framework: str
    model: str
    tp: int
    ep: int
    extra_server_args: str
    pd: dict


def _strip_dp_for_compute_bound_profile(extra_server_args: str, pd: dict) -> str:
    """Drop DP-attention / dp-size from the shared and per-role server args so the capture runs compute-bound."""
    for pd_key in ("pd_prefill_extra_args", "pd_decode_extra_args"):
        if pd.get(pd_key):
            pd[pd_key] = _strip_dp_parallel_flags(str(pd[pd_key]))
    log.info(
        "restart_server_for_round: compute-bound profile override active "
        "(%s=1) — stripped DP-attention/dp-size for this capture",
        _COMPUTE_BOUND_PROFILE_ENV,
    )
    return _strip_dp_parallel_flags(extra_server_args)


def _resolve_round(
    framework: str | None,
    model_path: str | None,
    tp: int | None,
    ep: int | None,
    extra_server_args: str,
) -> _ResolvedRound:
    """Resolve and vet everything the next round's server launches with."""
    fw, mdl, tp_int, ep_int = _resolve_round_args(framework, model_path, tp, ep)
    pd = _resolve_pd_args(tp_int=tp_int, ep_int=ep_int)
    # Apply Magpie's sglang DEFAULT_ARGS only for sglang (vllm has its own).
    if fw == "sglang":
        extra_server_args = _merge_sglang_defaults(extra_server_args)
    # Compute-bound profile override (multi-node only; already gated by the is_multi_node() no-op above).
    if os.environ.get(_COMPUTE_BOUND_PROFILE_ENV, "").strip() == "1":
        extra_server_args = _strip_dp_for_compute_bound_profile(extra_server_args, pd)
    try:
        validate_server_args(extra_server_args, context="restart_server_for_round")
    except ServerArgsRejected as exc:
        raise ServerRestartFailed(str(exc)) from exc
    return _ResolvedRound(fw, mdl, tp_int, ep_int, extra_server_args, pd)


_RESTART_LOCK: "asyncio.Lock | None" = None


def _get_restart_lock() -> "asyncio.Lock":
    """Serialize multi-node server restart+wait (single shared cluster server)."""
    global _RESTART_LOCK
    if _RESTART_LOCK is None:
        _RESTART_LOCK = asyncio.Lock()
    return _RESTART_LOCK


def _uses_aiter(
    extra_server_args: str,
    pd: dict | None,
    extra_env: dict[str, str] | None,
) -> bool:
    """True when this restart requests AMD aiter kernels (MoE/attention)."""
    parts = [extra_server_args or ""]
    if pd:
        parts.append(str(pd.get("pd_prefill_extra_args") or ""))
        parts.append(str(pd.get("pd_decode_extra_args") or ""))
    if "aiter" in " ".join(parts).lower():
        return True
    if extra_env and str(extra_env.get("SGLANG_USE_AITER", "")).strip() in {"1", "true", "True"}:
        return True
    return False


def _poll_timeout_env_s() -> int:
    """Launch-driver poll budget for one restart."""
    return int(
        os.environ.get(
            "HYPERLOOM_MN_POLL_TIMEOUT_S",
            str(DEFAULT_HEALTH_TIMEOUT_S),
        )
        or DEFAULT_HEALTH_TIMEOUT_S
    )


def _health_wait_env_s() -> int:
    """Post-launch /health budget for one restart."""
    return int(
        os.environ.get(
            "HYPERLOOM_MN_HEALTH_WAIT_S",
            str(DEFAULT_HEALTH_TIMEOUT_S),
        )
        or DEFAULT_HEALTH_TIMEOUT_S
    )


def _prepare_trace_dir(torch_profiler_dir: str) -> None:
    """Create the directory the restarted server writes this round's torch traces into."""
    if not torch_profiler_dir:
        return
    try:
        Path(torch_profiler_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ServerRestartFailed(f"cannot mkdir torch_profiler_dir {torch_profiler_dir!r}: {exc}") from exc


def _round_control_env(
    torch_profiler_dir: str,
    extra_env: dict[str, str] | None,
    unset_env: list[str] | tuple[str, ...] | set[str] | None,
) -> dict[str, str | None]:
    """The control variables this round's launcher reads out of the environment."""
    forwarded = (
        json.dumps(filter_forward_env({str(k): str(v) for k, v in extra_env.items()}, warn_on_drop=True))
        if extra_env
        else None
    )
    unset_keys = [str(k).strip() for k in (unset_env or []) if str(k).strip()]
    return {
        _PROFILE_TRACE_DIR_ENV: torch_profiler_dir or None,
        _EXTRA_FWD_ENV: forwarded,
        _UNSET_FWD_ENV: json.dumps(unset_keys) if unset_keys else None,
    }


async def _fan_out_tracelens_patch(torch_profiler_dir: str) -> None:
    """Apply the TraceLens SGLang patch set on every node before this restart (fail-soft)."""
    from ._server_patcher import resolve_sglang_patch_set
    from ._workload_envs import _tracelens_patch_enabled

    patch_set = resolve_sglang_patch_set()
    if patch_set == "graph-capture" and not torch_profiler_dir:
        log.info("restart_server_for_round: not a profiling round; skipping sglang_gc_patch fan-out")
        return
    if not (_tracelens_patch_enabled() and os.environ.get("TRACELENS_ROOT", "").strip()):
        return
    try:
        from hyperloom.inference_optimizer.multi_node.cli import cmd_apply_tracelens_patch

        patch_ns = argparse.Namespace(
            tracelens_root=os.environ.get("TRACELENS_ROOT", "").strip(),
            sglang_version_pin=os.environ.get(
                "HYPERLOOM_SGLANG_VERSION_PIN",
                "",
            ).strip()
            or None,
            patch_set=patch_set,
            print_logs=False,
            poll_interval=_POLL_INTERVAL_S,
            poll_timeout=_poll_timeout_env_s(),
        )
        patch_rc = await asyncio.to_thread(cmd_apply_tracelens_patch, patch_ns)
        if patch_rc != 0:
            log.warning(
                "restart_server_for_round: TraceLens SGLang patch fan-out "
                "returned rc=%d; proceeding with restart (trace may be "
                "unannotated, or CUDA-graph capture may IndexError on "
                ">= 0.5.18, until patches succeed)",
                patch_rc,
            )
    except Exception as exc:  # noqa: BLE001 - fail-soft envelope
        log.warning(
            "restart_server_for_round: TraceLens patch fan-out raised (%s); proceeding with restart (fail-soft)",
            exc,
        )


@dataclass(frozen=True)
class _RestartWaits:
    """How long one restart may spend in the launch driver and in the post-launch /health gate."""

    poll_timeout_s: int
    health_wait_s: int


def _resolve_restart_waits(
    round_args: _ResolvedRound,
    extra_env: dict[str, str] | None,
    driver_poll_timeout_s: int,
) -> _RestartWaits:
    """Resolve this restart's budgets, widening them for aiter's cold JIT/autotune."""
    # Align launch-driver poll with /health wait for JIT-heavy MoE runs.
    poll_timeout_s = max(_poll_timeout_env_s(), driver_poll_timeout_s)
    health_wait_s = _health_wait_env_s()
    if "HYPERLOOM_MN_HEALTH_WAIT_S" in os.environ:
        return _RestartWaits(poll_timeout_s, health_wait_s)
    if not _uses_aiter(round_args.extra_server_args, round_args.pd, extra_env):
        return _RestartWaits(poll_timeout_s, health_wait_s)
    aiter_wait = int(os.environ.get("HYPERLOOM_MN_HEALTH_WAIT_AITER_S", "1800") or 1800)
    if aiter_wait <= health_wait_s:
        return _RestartWaits(poll_timeout_s, health_wait_s)
    log.info(
        "restart_server_for_round: aiter kernels detected; widening "
        "worker /health wait %ds -> %ds for cold JIT/autotune "
        "(HYPERLOOM_MN_HEALTH_WAIT_AITER_S)",
        health_wait_s,
        aiter_wait,
    )
    return _RestartWaits(max(poll_timeout_s, aiter_wait), aiter_wait)


def _restart_namespace(round_args: _ResolvedRound, poll_timeout_s: int) -> argparse.Namespace:
    """The launch-driver arguments for one restart attempt."""
    pd = round_args.pd
    return argparse.Namespace(
        framework=round_args.framework,
        model=round_args.model,
        tp=round_args.tp,
        ep=round_args.ep,
        extra_args=round_args.extra_server_args or "",
        pid_file=None,
        log_file=None,
        no_wait_health=False,
        print_logs=False,
        poll_interval=_POLL_INTERVAL_S,
        poll_timeout=poll_timeout_s,
        # PD knobs; aggregated mode passes only pd_mode.
        pd_mode=pd.get("pd_mode", "aggregated"),
        pd_prefill_nodes=pd.get("pd_prefill_nodes", 0),
        pd_decode_nodes=pd.get("pd_decode_nodes", 0),
        pd_prefill_tp=pd.get("pd_prefill_tp", 0),
        pd_decode_tp=pd.get("pd_decode_tp", 0),
        pd_transfer_backend=pd.get("pd_transfer_backend", ""),
        pd_ib_device=pd.get("pd_ib_device", ""),
        # Per-role EP / extra-args (disaggregated only; 0 / "" => fall back to the shared ep / extra_args in
        # the CLI fan-out).
        pd_prefill_ep=pd.get("pd_prefill_ep", 0),
        pd_decode_ep=pd.get("pd_decode_ep", 0),
        pd_prefill_extra_args=pd.get("pd_prefill_extra_args", ""),
        pd_decode_extra_args=pd.get("pd_decode_extra_args", ""),
        pd_bootstrap_port=8998,
        pd_vllm_router_cmd="",
    )


def _log_restart_banner(round_args: _ResolvedRound, torch_profiler_dir: str) -> None:
    """Announce the round this restart is launching."""
    from ._multi_node_env import log_mn_banner

    pd = round_args.pd
    log_mn_banner(
        "server_restart",
        log,
        framework=round_args.framework,
        tp=round_args.tp,
        ep=round_args.ep,
        pd_mode=pd.get("pd_mode"),
        trace_dir=torch_profiler_dir or "",
    )
    log.info(
        "restart_server_for_round: framework=%s tp=%d ep=%d pd_mode=%s "
        "pd_prefill=%dx tp%d pd_decode=%dx tp%d backend=%r ib=%r "
        "extra_args=%r torch_profiler_dir=%r",
        round_args.framework,
        round_args.tp,
        round_args.ep,
        pd.get("pd_mode"),
        pd.get("pd_prefill_nodes", 0),
        pd.get("pd_prefill_tp", 0),
        pd.get("pd_decode_nodes", 0),
        pd.get("pd_decode_tp", 0),
        pd.get("pd_transfer_backend", ""),
        pd.get("pd_ib_device", ""),
        round_args.extra_server_args,
        torch_profiler_dir,
    )


async def _reclaim_remote_vram() -> None:
    """Best-effort remote kill-inference, so VRAM a crashed prior server still pins is free for the retry."""
    try:
        from hyperloom.inference_optimizer.multi_node.cli import kill_inference_for_kernel_agent_best_effort

        await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)
    except Exception as reclaim_exc:  # noqa: BLE001 - reclaim is best-effort
        log.warning(
            "restart_server_for_round: remote kill-inference reclaim raised (%s); retrying restart anyway",
            reclaim_exc,
        )


async def _restart_with_reclaim_retry(
    round_args: _ResolvedRound,
    *,
    extra_env: dict[str, str] | None,
    torch_profiler_dir: str,
    force_full_restart: bool,
) -> None:
    """Launch this round's server, reclaiming remote VRAM and retrying once when the first attempt fails."""
    # Local import to keep httpx out of the single-node import path.
    from hyperloom.inference_optimizer.multi_node.cli import cmd_restart_server, _resolve_poll_timeout_s

    waits = _resolve_restart_waits(round_args, extra_env, _resolve_poll_timeout_s())
    ns = _restart_namespace(round_args, waits.poll_timeout_s)
    _log_restart_banner(round_args, torch_profiler_dir)

    # One kill+launch attempt + post-launch /health wait.
    async def _restart_and_wait(force_full: bool) -> None:
        """Run one restart attempt and wait for /health readiness."""
        with _scoped_env({_RESUME_RUNNING_ENV: "0"} if force_full else {}):
            try:
                rc = await asyncio.to_thread(cmd_restart_server, ns)
            except Exception as exc:
                raise ServerRestartFailed(f"cmd_restart_server raised: {exc!r}") from exc

        if rc != 0:
            raise ServerRestartFailed(
                f"cmd_restart_server returned non-zero rc={rc} "
                f"(framework={round_args.framework} tp={round_args.tp} "
                f"extra_args={round_args.extra_server_args!r})"
            )
        try:
            await wait_for_restarted_server(waits.health_wait_s)
        except ServerRestartFailed as exc:
            collect_worker_server_logs(_read_state() or {}, str(exc))
            raise
        except Exception as exc:
            collect_worker_server_logs(_read_state() or {}, repr(exc))
            raise ServerRestartFailed(f"post-launch /health wait raised: {exc!r}") from exc

    try:
        await _restart_and_wait(force_full_restart)
    except ServerRestartFailed as first_exc:
        # C — multi-node VRAM reclaim before exactly one retry.
        if not env_flag(_MN_RESTART_RECLAIM_RETRY_ENV, default=True):
            raise
        log.warning(
            "restart_server_for_round: restart failed (%s); attempting best-effort remote kill-inference + one retry",
            first_exc,
        )
        await _reclaim_remote_vram()
        try:
            await _restart_and_wait(force_full=True)
        except ServerRestartFailed as retry_exc:
            raise retry_exc from first_exc


async def restart_server_for_round(
    *,
    extra_server_args: str = "",
    extra_env: dict[str, str] | None = None,
    unset_env: list[str] | tuple[str, ...] | set[str] | None = None,
    torch_profiler_dir: str = "",
    framework: str | None = None,
    model_path: str | None = None,
    tp: int | None = None,
    ep: int | None = None,
    force_full_restart: bool = False,
) -> None:
    """Restart the multi-node inference server for the next Magpie round."""
    if not is_multi_node():
        return

    # External mode without SSH control: no SaFE-managed pods to restart -- the benchmark runs against the
    # already-running server.
    from hyperloom.inference_optimizer.multi_node._internal.external_state import (
        external_has_server_control,
        external_service_url,
    )

    if external_service_url() and not external_has_server_control():
        log.info(
            "restart_server_for_round: external service URL (benchmark-only, no SSH control); skipping server restart"
        )
        return

    round_args = _resolve_round(framework, model_path, tp, ep, extra_server_args)

    async with _get_restart_lock():
        _prepare_trace_dir(torch_profiler_dir)
        with _scoped_env(_round_control_env(torch_profiler_dir, extra_env, unset_env)):
            await _fan_out_tracelens_patch(torch_profiler_dir)
            await _restart_with_reclaim_retry(
                round_args,
                extra_env=extra_env,
                torch_profiler_dir=torch_profiler_dir,
                force_full_restart=force_full_restart,
            )


# Infera frontend profiling API (infera.server --enable-profiling).


async def trigger_infera_engine_profile(
    action: str,
    body: dict | None = None,
) -> None:
    """Drive torch profiling on every Infera worker via the frontend fan-out API."""
    if not is_multi_node():
        return
    state = _read_state() or {}
    if str(state.get("backend") or "").strip().lower() != "infera":
        return
    service_url = str(state.get("service_url") or "").strip().rstrip("/")
    if not service_url:
        log.warning(
            "trigger_infera_engine_profile(%s): no service_url in state",
            action,
        )
        return
    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover
        log.warning("httpx unavailable; cannot trigger infera profiling")
        return
    route = "start" if action == "start" else "stop"
    url = f"{service_url}/v1/admin/profile/{route}"
    payload = dict(body or {}) if action == "start" else {}
    if action == "start":
        client_timeout = 30.0
    else:
        client_timeout = float(os.environ.get("HYPERLOOM_MN_STOP_PROFILE_TIMEOUT_S", "600") or 600)
    async with _httpx.AsyncClient(timeout=client_timeout) as client:
        try:
            resp = await client.post(url, json=payload)
            log.info(
                "infera profile %s -> %s HTTP %d",
                route,
                url,
                resp.status_code,
            )
        except _httpx.HTTPError as exc:
            log.warning("infera profile %s -> %s failed (%s); continuing", route, url, exc)


__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_S",
    "ServerRestartFailed",
    "_merge_sglang_defaults",
    "restart_server_for_round",
]
