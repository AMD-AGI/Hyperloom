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
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from hyperloom.common.timeutil import format_exc_brief
from hyperloom.common.env import EnvValueError, env_flag
from hyperloom.inference_optimizer import multi_node
from hyperloom.inference_optimizer.multi_node._internal.env_safety import filter_forward_env
from hyperloom.inference_optimizer.multi_node._internal.server_args_safety import (
    ServerArgsRejected,
    validate_server_args,
)
from ._multi_node_env import _read_state, is_multi_node
from ._multi_node_restart_settings import RestartSettings

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


class ServerRestartFailed(RuntimeError):
    """Raised when the per-round multi-node server restart did not succeed."""


def _state_or_env(state: dict, state_key: str, env_key: str, default: str = "") -> str:
    return str(state.get(state_key) or os.environ.get(env_key, "") or default).strip()


def _state_or_env_int(state: dict, state_key: str, env_key: str) -> int:
    for raw in (state.get(state_key), os.environ.get(env_key, "")):
        if raw:
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return 0


def _validate_pd_split(state_nodes: int, pn: int, dn: int, ptp: int, dtp: int) -> None:
    if pn <= 0 or dn <= 0:
        raise ServerRestartFailed(
            f"pd_mode=disaggregated requires pd_prefill_nodes>0 and pd_decode_nodes>0; got pn={pn} dn={dn}"
        )
    if state_nodes > 0 and pn + dn != state_nodes:
        raise ServerRestartFailed(
            f"pd_prefill_nodes ({pn}) + pd_decode_nodes ({dn}) must equal total nodes ({state_nodes})"
        )
    if ptp <= 0 or dtp <= 0:
        raise ServerRestartFailed(f"pd_prefill_tp ({ptp}) and pd_decode_tp ({dtp}) must be positive")


def _resolve_pd_args(*, tp_int: int) -> dict:
    """Resolve PD knobs with state.json + env fallback."""
    state = _read_state()
    mode = _state_or_env(state, "last_restart_pd_mode", "PD_MODE", "aggregated").lower()
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

    pn = _state_or_env_int(state, "last_restart_pd_prefill_nodes", "PD_PREFILL_NODES")
    dn = _state_or_env_int(state, "last_restart_pd_decode_nodes", "PD_DECODE_NODES")
    # Resume fallback: the restart path launches with ``pn or len(pods)`` but persists the raw arg (often 0), so a
    # resume that also lost the ``$PD_*_NODES`` env would leave pn/dn at 0 and wrongly fail the disaggregated gate
    # below (e.g. auto-roofline after resume).
    if pn <= 0:
        pn = len(state.get("prefill_pod_ips") or state.get("prefill_pods") or [])
    if dn <= 0:
        dn = len(state.get("decode_pod_ips") or state.get("decode_pods") or [])
    ptp = _state_or_env_int(state, "last_restart_pd_prefill_tp", "PD_PREFILL_TP") or tp_int
    dtp = _state_or_env_int(state, "last_restart_pd_decode_tp", "PD_DECODE_TP") or tp_int
    _validate_pd_split(state_nodes, pn, dn, ptp, dtp)

    out.update(
        {
            "pd_prefill_nodes": pn,
            "pd_decode_nodes": dn,
            "pd_prefill_tp": ptp,
            "pd_decode_tp": dtp,
            "pd_transfer_backend": _state_or_env(state, "last_restart_pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            "pd_ib_device": _state_or_env(state, "last_restart_pd_ib_device", "PD_IB_DEVICE"),
            # Per-role EP / extra server args; 0 / "" falls back to the shared --ep / --extra-args.
            "pd_prefill_ep": _state_or_env_int(state, "last_restart_pd_prefill_ep", "PD_PREFILL_EP"),
            "pd_decode_ep": _state_or_env_int(state, "last_restart_pd_decode_ep", "PD_DECODE_EP"),
            "pd_prefill_extra_args": _state_or_env(
                state, "last_restart_pd_prefill_extra_args", "PD_PREFILL_EXTRA_ARGS"
            ),
            "pd_decode_extra_args": _state_or_env(state, "last_restart_pd_decode_extra_args", "PD_DECODE_EXTRA_ARGS"),
        }
    )
    return out


@dataclass(frozen=True)
class RoundServer:
    """The server one multi-node round restarts; unset identity fields fall back to state.json, then env."""

    extra_server_args: str = ""
    framework: str | None = None
    model_path: str | None = None
    tp: int | None = None
    ep: int | None = None

    @classmethod
    def from_env(cls, extra_server_args: str, model_path: str | None) -> RoundServer:
        """This round's args and model on the framework, TP and EP the launch environment names."""
        return cls(
            extra_server_args,
            framework=os.environ.get("FRAMEWORK") or None,
            model_path=model_path,
            tp=int(os.environ.get("TP") or 0) or None,
            ep=int(os.environ.get("EP") or 0) or None,
        )


def _resolve_round_args(server: RoundServer) -> tuple[str, str, int, int]:
    """Resolve (framework, model, tp, ep) for the restart, with state fallback."""
    state = _read_state()
    fw = (server.framework or _state_or_env(state, "last_restart_framework", "FRAMEWORK", "sglang")).strip().lower()
    mdl = (server.model_path or _state_or_env(state, "last_restart_model", "MODEL_PATH")).strip()
    try:
        tp_int = int(server.tp if server.tp is not None else _state_or_env(state, "last_restart_tp", "TP", "0"))
    except (TypeError, ValueError):
        tp_int = 0
    try:
        ep_int = int(server.ep if server.ep is not None else _state_or_env(state, "last_restart_ep", "EP", "1"))
    except (TypeError, ValueError):
        ep_int = 1
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


# Per-variant env overrides -> forwarded to the SSH-launched sglang via
# ``multi_node/commands/infera.py::_collect_forward_env`` (reads these control envs).
def _restart_env_scope(extra_env: dict[str, str] | None, unset_env, torch_profiler_dir: str):
    forwarded = json.dumps(filter_forward_env({str(k): str(v) for k, v in extra_env.items()})) if extra_env else None
    unset_keys = [str(k).strip() for k in (unset_env or []) if str(k).strip()]
    return _scoped_env(
        {
            _PROFILE_TRACE_DIR_ENV: torch_profiler_dir or None,
            _EXTRA_FWD_ENV: forwarded,
            _UNSET_FWD_ENV: json.dumps(unset_keys) if unset_keys else None,
        }
    )


# Multi-node TraceLens SGLang patch fan-out (fail-soft). The patch set (roofline vs sglang_gc_patch) follows
# this controller's shape mode; sglang_gc_patch only matters while capturing, so it waits for a profiling round.
async def _apply_tracelens_patch_fanout(patch_set: str, torch_profiler_dir: str, poll_timeout_s: int) -> None:
    from ._profile_setup import _tracelens_patch_enabled

    if patch_set == "graph-capture" and not torch_profiler_dir:
        log.info("restart_server_for_round: not a profiling round; skipping sglang_gc_patch fan-out")
        return
    if not _tracelens_patch_enabled() or not os.environ.get("TRACELENS_ROOT", "").strip():
        return
    import httpx

    try:
        from hyperloom.inference_optimizer.multi_node.cli import cmd_apply_tracelens_patch

        patch_ns = argparse.Namespace(
            tracelens_root=os.environ.get("TRACELENS_ROOT", "").strip(),
            sglang_version_pin=os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "").strip() or None,
            patch_set=patch_set,
            print_logs=False,
            poll_interval=_POLL_INTERVAL_S,
            poll_timeout=poll_timeout_s,
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
    except (RuntimeError, ValueError, OSError, KeyError, httpx.HTTPError, httpx.InvalidURL) as exc:
        log.warning(
            "restart_server_for_round: TraceLens patch fan-out raised (%s); proceeding with restart (fail-soft)",
            exc,
        )


def _read_restart_settings() -> RestartSettings:
    """The restart's settings; a malformed one fails the restart before it touches the cluster."""
    try:
        return RestartSettings.from_env()
    except EnvValueError as exc:
        raise ServerRestartFailed(f"invalid multi-node restart setting: {exc}") from exc


def _health_wait_budget(
    settings: RestartSettings, extra_server_args: str, pd: dict, extra_env: dict[str, str] | None
) -> tuple[int, int]:
    poll_timeout_s, health_wait_s = settings.poll_timeout_s, settings.health_wait_s
    # aiter kernels JIT-compile + autotune on first use (server log: "not found tuned config in
    # /tmp/aiter_configs"); a cold compile can exceed the default 900s gate and false-fail an
    # otherwise-healthy variant (the doomed attempt then burns a reclaim+retry cycle before the now-warm
    # relaunch succeeds).
    if settings.health_wait_explicit or not _uses_aiter(extra_server_args, pd, extra_env):
        return poll_timeout_s, health_wait_s
    aiter_wait = settings.health_wait_aiter_s
    if aiter_wait <= health_wait_s:
        return poll_timeout_s, health_wait_s
    log.info(
        "restart_server_for_round: aiter kernels detected; widening "
        "worker /health wait %ds -> %ds for cold JIT/autotune "
        "(HYPERLOOM_MN_HEALTH_WAIT_AITER_S)",
        health_wait_s,
        aiter_wait,
    )
    return max(poll_timeout_s, aiter_wait), aiter_wait


def _restart_namespace(
    round_args: tuple[str, str, int, int], extra_server_args: str, poll_timeout_s: int, pd: dict
) -> argparse.Namespace:
    fw, mdl, tp_int, ep_int = round_args
    return argparse.Namespace(
        framework=fw,
        model=mdl,
        tp=tp_int,
        ep=ep_int,
        extra_args=extra_server_args or "",
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


async def restart_server_for_round(
    server: RoundServer,
    *,
    extra_env: dict[str, str] | None = None,
    unset_env: list[str] | tuple[str, ...] | set[str] | None = None,
    torch_profiler_dir: str = "",
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

    round_args = _resolve_round_args(server)
    fw, mdl, tp_int, ep_int = round_args
    pd = _resolve_pd_args(tp_int=tp_int)

    # PD-disaggregated x EP cross-check: ep must not exceed either group's TP.
    min_grp_tp = min(pd.get("pd_prefill_tp", 0), pd.get("pd_decode_tp", 0))
    if pd["pd_mode"] == "disaggregated" and ep_int > 1 and ep_int > min_grp_tp:
        raise ServerRestartFailed(
            f"ep={ep_int} > min(pd_prefill_tp={pd['pd_prefill_tp']}, "
            f"pd_decode_tp={pd['pd_decode_tp']})={min_grp_tp}; "
            "lower --ep or raise the smaller per-group TP."
        )

    # Apply Magpie's sglang DEFAULT_ARGS only for sglang (vllm has its own).
    extra_server_args = _merge_sglang_defaults(server.extra_server_args) if fw == "sglang" else server.extra_server_args

    # Compute-bound profile override (multi-node only; already gated by the is_multi_node() no-op above).
    if os.environ.get(_COMPUTE_BOUND_PROFILE_ENV, "").strip() == "1":
        extra_server_args = _strip_dp_parallel_flags(extra_server_args)
        for _pd_key in ("pd_prefill_extra_args", "pd_decode_extra_args"):
            if pd.get(_pd_key):
                pd[_pd_key] = _strip_dp_parallel_flags(str(pd[_pd_key]))
        log.info(
            "restart_server_for_round: compute-bound profile override active "
            "(%s=1) — stripped DP-attention/dp-size for this capture",
            _COMPUTE_BOUND_PROFILE_ENV,
        )

    try:
        validate_server_args(extra_server_args, context="restart_server_for_round")
    except ServerArgsRejected as exc:
        raise ServerRestartFailed(str(exc)) from exc
    settings = _read_restart_settings()

    async with _get_restart_lock():
        if torch_profiler_dir:
            try:
                Path(torch_profiler_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ServerRestartFailed(f"cannot mkdir torch_profiler_dir {torch_profiler_dir!r}: {exc}") from exc
        with _restart_env_scope(extra_env, unset_env, torch_profiler_dir):
            from ._server_patcher import resolve_sglang_patch_set

            await _apply_tracelens_patch_fanout(resolve_sglang_patch_set(), torch_profiler_dir, settings.poll_timeout_s)

            # Local import to keep httpx out of the single-node import path.
            from hyperloom.inference_optimizer.multi_node.cli import cmd_restart_server

            # Align launch-driver poll with /health wait for JIT-heavy MoE runs.
            poll_timeout_s, health_wait_s = _health_wait_budget(settings, extra_server_args, pd, extra_env)
            ns = _restart_namespace(round_args, extra_server_args, poll_timeout_s, pd)
            from ._multi_node_env import log_mn_banner

            log_mn_banner(
                "server_restart",
                log,
                framework=fw,
                tp=tp_int,
                ep=ep_int,
                pd_mode=pd.get("pd_mode"),
                trace_dir=torch_profiler_dir or "",
            )
            log.info(
                "restart_server_for_round: framework=%s tp=%d ep=%d pd_mode=%s "
                "pd_prefill=%dx tp%d pd_decode=%dx tp%d backend=%r ib=%r "
                "extra_args=%r torch_profiler_dir=%r",
                fw,
                tp_int,
                ep_int,
                pd.get("pd_mode"),
                pd.get("pd_prefill_nodes", 0),
                pd.get("pd_prefill_tp", 0),
                pd.get("pd_decode_nodes", 0),
                pd.get("pd_decode_tp", 0),
                pd.get("pd_transfer_backend", ""),
                pd.get("pd_ib_device", ""),
                extra_server_args,
                torch_profiler_dir,
            )

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
                        f"(framework={fw} tp={tp_int} extra_args={extra_server_args!r})"
                    )

                # cmd_restart_server returns when actors are spawned, but a cold MoE weight-load can need 20-30 min
                # before /health flips; poll it here so the downstream baseline doesn't fire against a not-yet-ready
                # server.
                try:
                    # PD restart: ensure BOTH prefill+decode legs are /health-ready (mooncake init done) before the
                    # frontend completions probe, so its grace does not expire against a half-ready pair.
                    await _wait_for_workers_ready_async(settings, timeout_s=health_wait_s)
                    await _wait_for_server_health_async(settings, timeout_s=health_wait_s)
                    # The reachable /health above can be a pod-pinned head address; the benchmark dials the published
                    # ClusterIP Service, whose endpoints lag readiness after a restart.
                    await _wait_for_published_service_ready_async(settings)
                except ServerRestartFailed as exc:
                    _collect_worker_server_logs(_read_state(), str(exc))
                    raise
                except Exception as exc:
                    _collect_worker_server_logs(_read_state(), repr(exc))
                    raise ServerRestartFailed(f"post-launch /health wait raised: {exc!r}") from exc

            try:
                await _restart_and_wait(force_full_restart)
            except ServerRestartFailed as first_exc:
                # C — multi-node VRAM reclaim before exactly one retry.
                if not env_flag(_MN_RESTART_RECLAIM_RETRY_ENV, default=True):
                    raise
                log.warning(
                    "restart_server_for_round: restart failed (%s); attempting "
                    "best-effort remote kill-inference + one retry",
                    first_exc,
                )
                import httpx

                try:
                    from hyperloom.inference_optimizer.multi_node.cli import kill_inference_for_kernel_agent_best_effort

                    await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)
                except (ValueError, KeyError, httpx.HTTPError, httpx.InvalidURL) as reclaim_exc:
                    log.warning(
                        "restart_server_for_round: remote kill-inference reclaim raised (%s); retrying restart anyway",
                        reclaim_exc,
                    )
                try:
                    await _restart_and_wait(force_full=True)
                except ServerRestartFailed as retry_exc:
                    raise retry_exc from first_exc


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


async def _probe_health(client: Any, health_url: str) -> tuple[bool, str]:
    import httpx as _httpx

    try:
        resp = await client.get(health_url)
    except _httpx.HTTPError as exc:
        return False, format_exc_brief(exc, limit=120)
    if resp.status_code == 200:
        return True, ""
    return False, f"http_status={resp.status_code}"


async def _probe_model_ids(client: Any, models_url: str) -> tuple[list[Any], str]:
    import httpx as _httpx

    try:
        resp = await client.get(models_url)
    except _httpx.HTTPError as exc:
        return [], f"models_probe {format_exc_brief(exc, limit=80)}"
    if resp.status_code != 200:
        return [], f"models_http_status={resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        data = {}
    models = data.get("data") if isinstance(data, dict) else None
    return (models if isinstance(models, list) else []), ""


async def _probe_completion_tokens(
    client: Any, completions_url: str, models: list[Any], max_tokens: int
) -> tuple[int, str]:
    import httpx as _httpx

    model_id = str(models[0].get("id") or "") if isinstance(models[0], dict) else ""
    if not model_id:
        return -1, "completion_probe: no model id"
    try:
        resp = await client.post(
            completions_url,
            json={
                "model": model_id,
                "prompt": "hi",
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            },
        )
    except _httpx.HTTPError as exc:
        return -1, f"completion_probe {format_exc_brief(exc, limit=80)}"
    if resp.status_code != 200:
        return -1, f"completion_probe_http={resp.status_code}"
    try:
        return multi_node.generated_tokens(resp.json()), ""
    except ValueError:
        return 0, ""


def _completions_stalled_too_long(*, elapsed: int, models_ready_at: int | None, grace_s: int) -> bool:
    if grace_s <= 0 or models_ready_at is None:
        return False
    return (elapsed - models_ready_at) > grace_s


def _models_empty_too_long(
    *,
    elapsed: int,
    health_ok_at: int | None,
    models_ready_at: int | None,
    grace_s: int,
) -> bool:
    """Whether to fast-fail because /v1/models never registered any worker."""
    if grace_s <= 0 or health_ok_at is None or models_ready_at is not None:
        return False
    return (elapsed - health_ok_at) > grace_s


def _worker_detokenizer_wedged(tail: str) -> bool:
    """Best-effort: True when a worker's server-log ``tail`` shows a persistent detokenizer wedge -- weights are loaded but the detokenizer never heartbeats, so the engine's /health never flips and the HTTP port never binds."""
    import re as _re

    # >=15 recent markers ~= 30s+ of continuous stall (marker ~every 2s).
    return len(_re.findall(r"Health check failed.*detokenizer", tail, _re.IGNORECASE)) >= 15


def _worker_startup_crashed(tail: str) -> str | None:
    """Best-effort: return a short reason when a worker's server-log ``tail`` shows a NON-RECOVERABLE startup failure (so /health will never flip), else None."""
    import re as _re

    patterns = (
        r"error: unrecognized arguments:.*",
        r"[A-Za-z0-9_./-]*: error: .*",  # argparse "<prog>: error: ..."
        r"error: the following arguments are required:.*",
        r"error: argument .*",
        # Worker subprocess died BEFORE ever reporting ready -- SIGKILL/OOM (exit code -9), non-zero exit, or an
        # explicit early-exit RuntimeError (e.g. ``sglang subprocess exited with code -9 before reporting ready``).
        # /health will never come up for a dead engine, so fast-fail instead of burning the (aiter-widened) health
        # gate.
        r".*exited with code -?\d+ before reporting ready.*",
        r".*(sglang|engine|worker) (sub)?process exited with code -?\d+.*",
        r"(?i).*(cuda|hip|rocm|torch)[^\n]{0,40}out of memory.*",
        r"(?i).*torch\.OutOfMemoryError.*",
    )
    for pat in patterns:
        m = _re.search(pat, tail)
        if m:
            return m.group(0).strip()[:200]
    return None


def _worker_health_targets(state: dict) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    targets: list[tuple[str, str, str]] = []
    for role_key in ("prefill_pods", "decode_pods", "worker_pods"):
        for pod in state.get(role_key) or []:
            if not isinstance(pod, dict):
                continue
            ip = str(pod.get("podIP") or "").strip()
            if not ip or ip in seen:
                continue
            seen.add(ip)
            targets.append((role_key.replace("_pods", ""), ip, str(pod.get("podId") or ip)))
    return targets


def _worker_fastfail_reason(
    pending: list[str], *, elapsed: int, crash_grace: int, wedge_grace: int, timeout_s: int
) -> str | None:
    if elapsed <= min(crash_grace, wedge_grace):
        return None
    # 64 KiB keeps the wedge rule about recent markers and keeps an earlier launch's crash, appended to the same log,
    # out of the window.
    tails = {ip: (_shared_worker_log_tail(ip, max_bytes=65536) or "").partition("\n")[2] for ip in pending}
    if elapsed > crash_grace:
        for ip, tail in tails.items():
            crashed = _worker_startup_crashed(tail)
            if crashed:
                return (
                    "worker " + ip + " server crashed on startup "
                    "(non-recoverable; /health will never come up): "
                    + crashed
                    + " -- fast-failed after "
                    + str(elapsed)
                    + "s instead of the full "
                    + str(timeout_s)
                    + "s gate"
                )
    if elapsed > wedge_grace:
        wedged = [ip for ip, tail in tails.items() if _worker_detokenizer_wedged(tail)]
        if wedged:
            return (
                "worker(s) detokenizer-wedged (weights loaded but "
                "detokenizer not heartbeating; /health never up): "
                + str(wedged)
                + " after "
                + str(elapsed)
                + "s; fast-failed "
                "instead of waiting the full " + str(timeout_s) + "s gate"
            )
    return None


async def _wait_for_workers_ready_async(settings: RestartSettings, timeout_s: int) -> None:
    """Wait for every prefill/decode worker's own /health to return 200 before the frontend serving probe runs."""
    import time as _t

    import httpx as _httpx

    port = settings.worker_port
    targets = _worker_health_targets(_read_state())
    if not targets:
        return
    started = _t.monotonic()
    ready: set[str] = set()
    # Detokenizer-wedge fast-fail: bail early on a wedged worker instead of burning the full timeout.
    wedge_grace = settings.worker_wedge_grace_s
    fastfail = env_flag("HYPERLOOM_MN_WORKER_WEDGE_FASTFAIL", default=True)
    # Startup-crash fast-fail: an argparse rejection / fatal exit is terminal, so bail within seconds instead of the
    # full gate.
    crash_grace = settings.worker_crash_grace_s
    log.info("waiting for %d worker(s) /health on port %d before frontend probe", len(targets), port)
    async with _httpx.AsyncClient(timeout=10.0) as client:
        while True:
            elapsed = int(_t.monotonic() - started)
            for role, ip, _pid in targets:
                if ip in ready:
                    continue
                try:
                    resp = await client.get("http://" + ip + ":" + str(port) + "/health")
                    if resp.status_code == 200:
                        ready.add(ip)
                        log.info(
                            "worker /health OK (%s %s) after %ds [%d/%d]", role, ip, elapsed, len(ready), len(targets)
                        )
                except _httpx.HTTPError:
                    pass
            pending = [ip for _r, ip, _p in targets if ip not in ready]
            if not pending:
                log.info(
                    "all %d worker(s) /health-ready after %ds; proceeding to frontend probe", len(targets), elapsed
                )
                return
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    "workers not /health-ready within "
                    + str(timeout_s)
                    + "s: "
                    + str(pending)
                    + " (a PD leg is still initializing, e.g. prefill "
                    "mooncake transfer engine)"
                )
            reason = (
                _worker_fastfail_reason(
                    pending, elapsed=elapsed, crash_grace=crash_grace, wedge_grace=wedge_grace, timeout_s=timeout_s
                )
                if fastfail
                else None
            )
            if reason:
                raise ServerRestartFailed(reason)
            await asyncio.sleep(settings.health_poll_s)


def _shared_server_log_dir() -> str:
    shared = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
    return shared if shared.startswith("/") and "$" not in shared else ""


def _shared_worker_log_tail(ip: str, max_bytes: int = 2_000_000) -> str | None:
    """Best-effort tail of a worker's shared-FS sglang server log."""
    import glob as _glob

    shared = _shared_server_log_dir()
    if not shared:
        return None
    matches = _glob.glob(os.path.join(shared, "mn_infera_server_" + str(ip) + "_r*.log"))
    if not matches:
        return None
    # Newest by mtime: the current (failed) launch overwrites/appends this file.
    path = max(matches, key=lambda p: os.path.getmtime(p))
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return "# source: " + path + "\n" + f.read().decode("utf-8", "replace")
    except OSError:
        return None


def _ssh_worker_log_tail(ip: str, port: int, remote: str, key_path: str, known_hosts: str) -> str:
    from hyperloom.inference_optimizer.multi_node import SshTarget, ssh_run

    cp = ssh_run(
        SshTarget(host=ip, key_path=key_path, known_hosts=known_hosts, port=port),
        "tail -c 2000000 " + remote + " 2>/dev/null",
        timeout=60,
    )
    body = cp.stdout or ""
    if cp.stderr:
        body += "\n--- ssh stderr ---\n" + cp.stderr
    return body


def _collect_worker_server_logs(state: dict, reason: str) -> None:
    """Best-effort: capture each prefill/decode pod sglang server log into ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/server_logs`` when a restart fails its health/serving probe, so a 503 / KV handoff / cold-JIT failure leaves a post-mortem after the pods are torn down."""
    import time as _t

    from hyperloom.inference_optimizer.multi_node import INFERA_POD_LOG_FILE

    sess = (
        os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR")
        or os.environ.get("INFERENCE_OPTIMIZER_SESSION_DIR")
        or ""
    )
    if not sess:
        return
    out_dir = os.path.join(sess, "server_logs")
    ts = _t.strftime("%Y%m%dT%H%M%SZ", _t.gmtime())
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        return
    key_path = state.get("ssh_key_path")
    known_hosts = state.get("ssh_known_hosts")
    remote = os.environ.get("HYPERLOOM_MN_SERVER_LOG_PATH", INFERA_POD_LOG_FILE)
    default_port = int(state.get("ssh_port") or 2233)
    for role in ("prefill", "decode", "worker"):
        for pod in state.get(role + "_pods") or []:
            ip = pod.get("podIP")
            if not ip:
                continue
            # 1) Shared-FS read (preferred: real trace, no SSH, survives teardown).
            body = _shared_worker_log_tail(str(ip))
            source = "shared-fs"
            if body is None and key_path and known_hosts:
                # 2) Fallback: SSH tail of the pod-local log.
                source = "ssh:" + str(remote)
                try:
                    body = _ssh_worker_log_tail(
                        str(ip), int(pod.get("sshPort") or default_port), remote, key_path, known_hosts
                    )
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    continue
            if body is None:
                continue
            try:
                dest = os.path.join(out_dir, ts + "_" + role + "_" + str(pod.get("podId") or ip) + ".log")
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write("# collected on restart failure (" + source + "): " + str(reason) + "\n")
                    fh.write(body)
                log.info("collected worker server log -> %s", dest)
            except OSError:
                continue


async def _wait_for_server_health_async(settings: RestartSettings, timeout_s: int) -> None:
    """Poll the multi_node service_url /health until 200 or timeout."""
    import time as _time

    import httpx as _httpx

    from hyperloom.inference_optimizer.multi_node._internal.external_state import reachable_service_url

    state = _read_state()
    service_url = reachable_service_url(state)
    if not service_url:
        log.warning("no service_url in state; skipping post-restart /health wait")
        return

    base = service_url.rstrip("/")
    health_url = base + "/health"
    started = _time.monotonic()
    last_err = ""
    log.info(
        "post-restart /health wait: url=%s timeout_s=%d poll_every_s=%d",
        health_url,
        timeout_s,
        settings.health_poll_s,
    )
    # Infera-ONLY serving-readiness extension. When it is on, also do a 1-token completion probe to confirm workers
    # actually serve traffic (Infera registers models in /v1/models before the worker is ready to accept requests;
    # this causes the first benchmark to get 503 "Model temporarily unavailable" for every request, surfacing as
    # `completed=0` -> `baseline_failed`).
    wait_model_ready = str(state.get("backend") or "").strip().lower() == "infera"
    health_ok_at = None
    models_ready_at = None
    consecutive_completion_ok = 0
    # require N consecutive successful completions before declaring ready
    completion_probe_required = settings.completion_probe_count
    # Probe with >1 token + ignore_eos so the request must traverse the decode leg (prefill alone can serve the first
    # token): a PD run with an unready or KV-broken decode then fails the probe instead of passing on a prefill-only
    # 200.
    completion_probe_tokens = settings.completion_probe_tokens
    completion_probe_min_tokens = settings.completion_probe_min_tokens
    # Fast-fail grace: if /health is up but /v1/models never registers within this many seconds, the GPU workers
    # crashed on launch (e.g. an unrecognized server flag) rather than slowly loading — bail early instead of burning
    # the full timeout_s.
    models_empty_grace_s = settings.models_empty_grace_s
    # Fast-fail when /v1/models is populated but /v1/completions never serves within this window (PD prefill<->decode
    # KV handoff broken on restart -> HTTP 503 every probe).
    completions_grace_s = settings.completions_grace_s
    async with _httpx.AsyncClient(timeout=15.0) as client:
        while True:
            elapsed = int(_time.monotonic() - started)
            health_ok, last_err = await _probe_health(client, health_url)
            if health_ok:
                if health_ok_at is None:
                    health_ok_at = elapsed
                    log.info(
                        "post-restart /health OK after %ds (url=%s)%s",
                        elapsed,
                        health_url,
                        "; now also waiting for /v1/models to be non-empty" if wait_model_ready else "",
                    )
                if not wait_model_ready:
                    return
                models, probe_err = await _probe_model_ids(client, base + "/v1/models")
                if not models:
                    if not probe_err:
                        consecutive_completion_ok = 0
                        probe_err = f"models_empty (health_ok_at={health_ok_at}s)"
                    last_err = probe_err
                else:
                    if models_ready_at is None:
                        models_ready_at = elapsed
                        log.info(
                            "post-restart /v1/models populated after %ds (n=%d, health_ok_at=%ds); now probing /v1/completions",
                            elapsed,
                            len(models),
                            health_ok_at,
                        )
                    gen_toks, probe_err = await _probe_completion_tokens(
                        client, base + "/v1/completions", models, completion_probe_tokens
                    )
                    if gen_toks >= completion_probe_min_tokens:
                        consecutive_completion_ok += 1
                        if consecutive_completion_ok >= completion_probe_required:
                            log.info(
                                "post-restart READY after %ds (n=%d, models_at=%ds, completion_ok_x%d, gen_tokens=%d)",
                                elapsed,
                                len(models),
                                models_ready_at,
                                consecutive_completion_ok,
                                gen_toks,
                            )
                            return
                        last_err = (
                            f"completion_probe ok x{consecutive_completion_ok}/"
                            f"{completion_probe_required} (gen_tokens={gen_toks})"
                        )
                    else:
                        # HTTP 200 but decode leg produced nothing (unready / broken PD KV handoff) — do NOT
                        # count as ready.
                        consecutive_completion_ok = 0
                        last_err = probe_err or (
                            f"completion_probe_zero_tokens gen={gen_toks} "
                            f"(need>={completion_probe_min_tokens}; decode leg not serving)"
                        )
            if _models_empty_too_long(
                elapsed=elapsed,
                health_ok_at=health_ok_at,
                models_ready_at=models_ready_at,
                grace_s=models_empty_grace_s,
            ):
                raise ServerRestartFailed(
                    f"/v1/models still empty {elapsed - health_ok_at}s after /health "
                    f"(> grace {models_empty_grace_s}s); GPU workers crashed on launch "
                    f"(url={base + '/v1/models'}, last_err={last_err})"
                )
            if _completions_stalled_too_long(
                elapsed=elapsed, models_ready_at=models_ready_at, grace_s=completions_grace_s
            ):
                raise ServerRestartFailed(
                    f"/v1/models populated at {models_ready_at}s but /v1/completions still "
                    f"not serving {elapsed - models_ready_at}s later (> grace {completions_grace_s}s); "
                    f"serving broken, likely PD KV handoff on restart "
                    f"(url={base + '/v1/completions'}, last_err={last_err})"
                )
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    f"server /health did not return 200 within {timeout_s}s (url={health_url}, last_err={last_err})"
                )
            if elapsed % 60 < settings.health_poll_s:
                log.info(
                    "post-restart /health still waiting t=%ds last_err=%s",
                    elapsed,
                    last_err,
                )
            await asyncio.sleep(settings.health_poll_s)


def _is_name_resolution_error(exc: BaseException) -> bool:
    """Whether an httpx/OS error was caused by DNS name resolution failing."""
    import socket as _socket

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, _socket.gaierror):
            return True
        text = str(cur).lower()
        if (
            "name or service not known" in text
            or "nodename nor servname" in text
            or "temporary failure in name resolution" in text
            or "getaddrinfo failed" in text
            or "no address associated with hostname" in text
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


async def _wait_for_published_service_ready_async(settings: RestartSettings) -> None:
    """Gate on the published ``service_url`` the benchmark dials, not only the reachable address :func:`_wait_for_server_health_async` proves."""
    import time as _time

    if settings.published_ready_s <= 0:
        log.info("post-restart published-service gate disabled (HYPERLOOM_MN_PUBLISHED_READY_S<=0)")
        return

    import httpx as _httpx

    from hyperloom.inference_optimizer.multi_node._internal.external_state import reachable_service_url

    state = _read_state() or {}
    if str(state.get("backend") or "").strip().lower() != "rayjob":
        return
    published = str(state.get("service_url") or "").strip().rstrip("/")
    reachable = reachable_service_url(state).rstrip("/")
    # Nothing new to prove without a rewrite: the reachable wait already covered this exact endpoint.
    if not published or published == reachable:
        return

    health_url = published + "/health"
    # Consecutive 200s required so a single endpoint-list flap does not pass us (override:
    # HYPERLOOM_MN_PUBLISHED_READY_OK_STREAK).
    required_ok = settings.published_ready_ok_streak
    ok_streak = 0
    # Skip only after this many CONSECUTIVE name-resolution failures, so a CoreDNS blip inside the readiness window
    # (which resolves on a later poll) does not disable the gate exactly when it is needed.
    dns_skip_after = settings.published_dns_skip_after
    dns_failures = 0
    started = _time.monotonic()
    last_err = ""
    log.info(
        "post-restart published-service /health wait: url=%s timeout_s=%d",
        health_url,
        settings.published_ready_s,
    )
    async with _httpx.AsyncClient(timeout=15.0) as client:
        while True:
            elapsed = int(_time.monotonic() - started)
            try:
                resp = await client.get(health_url)
                dns_failures = 0  # resolved + connected: not an unresolvable name
                if resp.status_code == 200:
                    ok_streak += 1
                    if ok_streak >= required_ok:
                        log.info(
                            "post-restart published-service /health OK after %ds (url=%s)",
                            elapsed,
                            health_url,
                        )
                        return
                else:
                    ok_streak = 0
                    last_err = f"http_status={resp.status_code}"
            except _httpx.HTTPError as exc:
                if _is_name_resolution_error(exc):
                    dns_failures += 1
                    if dns_failures >= dns_skip_after:
                        # Surface, don't degrade silently: to the caller this is indistinguishable from a pass, so a
                        # later completed=0 / magpie_nonzero_invalid_measurement has no other clue it was this skip.
                        log.warning(
                            "published service_url %s did not resolve from here after %d consecutive "
                            "attempts (~%ds); SKIPPING the published-service readiness gate. If this is "
                            "an in-cluster run this is a DNS problem, and a following benchmark "
                            "completed=0 / ECONNREFUSED likely traces back to this skip.",
                            health_url,
                            dns_failures,
                            dns_failures * settings.health_poll_s,
                        )
                        return
                    ok_streak = 0
                    last_err = f"name resolution failed ({dns_failures}/{dns_skip_after})"
                else:
                    dns_failures = 0  # reached the host: a real refuse/5xx, keep waiting
                    ok_streak = 0
                    last_err = format_exc_brief(exc, limit=120)
            if elapsed > settings.published_ready_s:
                raise ServerRestartFailed(
                    f"published service /health did not return 200 within {settings.published_ready_s}s "
                    f"(url={health_url}, last_err={last_err}); the endpoint the benchmark "
                    f"dials had no ready backend (post-restart readiness window did not close)"
                )
            await asyncio.sleep(settings.health_poll_s)


__all__ = [
    "RoundServer",
    "ServerRestartFailed",
    "_merge_sglang_defaults",
    "restart_server_for_round",
]
