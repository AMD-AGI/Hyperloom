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
import tempfile
from pathlib import Path

from ...loop.coordinator_helpers import format_exc_brief
from hyperloom.inference_optimizer.multi_node._internal.env_safety import filter_forward_env
from hyperloom.inference_optimizer.multi_node._internal.server_args_safety import (
    ServerArgsRejected,
    validate_server_args,
)
from ._multi_node_env import _read_state, is_multi_node

# Scoped env flag (set by the roofline compute-bound re-profile) that tells this restart to strip DP-attention /
# dp-size from prefill+decode+shared server args, so the profiled server runs single-rank full-batch (compute-bound).
_COMPUTE_BOUND_PROFILE_ENV = "HYPERLOOM_MN_PROFILE_COMPUTE_BOUND"

# Gate for the reclaim-and-retry path in ``restart_server_for_round``: when a restart fails, best-effort remote
# ``kill-inference`` (Infera SSH fan-out / RayJob Dashboard kill job) reclaims VRAM pinned by a crashed prior server,
# then exactly one forced fresh restart is retried.
_MN_RESTART_RECLAIM_RETRY_ENV = "HYPERLOOM_MN_RESTART_RECLAIM_RETRY"


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


def _resolve_pd_args(
    pd_mode: str | None,
    pd_prefill_nodes: int | None,
    pd_decode_nodes: int | None,
    pd_prefill_tp: int | None,
    pd_decode_tp: int | None,
    pd_transfer_backend: str | None,
    pd_ib_device: str | None,
    *,
    tp_int: int,
) -> dict:
    """Resolve PD knobs with state.json + env fallback."""
    state = _read_state()
    mode = (
        (pd_mode or state.get("last_restart_pd_mode") or os.environ.get("PD_MODE", "") or "aggregated").strip().lower()
    )
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

    def _intf(kw, sk, ek):
        """Resolve an int field from kwarg > state key > env var."""
        if kw is not None:
            return int(kw)
        v = state.get(sk)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        v = os.environ.get(ek, "")
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        return 0

    pn = _intf(pd_prefill_nodes, "last_restart_pd_prefill_nodes", "PD_PREFILL_NODES")
    dn = _intf(pd_decode_nodes, "last_restart_pd_decode_nodes", "PD_DECODE_NODES")
    # Resume fallback: the restart path launches with ``pn or len(pods)`` but persists the raw arg (often 0), so a
    # resume that also lost the ``$PD_*_NODES`` env would leave pn/dn at 0 and wrongly fail the disaggregated gate
    # below (e.g. auto-roofline after resume).
    if pn <= 0:
        pn = len(state.get("prefill_pod_ips") or state.get("prefill_pods") or [])
    if dn <= 0:
        dn = len(state.get("decode_pod_ips") or state.get("decode_pods") or [])
    ptp = _intf(pd_prefill_tp, "last_restart_pd_prefill_tp", "PD_PREFILL_TP") or tp_int
    dtp = _intf(pd_decode_tp, "last_restart_pd_decode_tp", "PD_DECODE_TP") or tp_int
    tb = (
        pd_transfer_backend
        or state.get("last_restart_pd_transfer_backend")
        or os.environ.get("PD_TRANSFER_BACKEND", "")
        or ""
    ).strip()
    ib = (pd_ib_device or state.get("last_restart_pd_ib_device") or os.environ.get("PD_IB_DEVICE", "") or "").strip()
    # Per-role EP / extra server args, resolved from state + env only (no kwarg). 0 / "" falls back to the shared --ep
    # / --extra-args.
    pep = _intf(None, "last_restart_pd_prefill_ep", "PD_PREFILL_EP")
    dep = _intf(None, "last_restart_pd_decode_ep", "PD_DECODE_EP")
    prefill_extra = (
        state.get("last_restart_pd_prefill_extra_args") or os.environ.get("PD_PREFILL_EXTRA_ARGS", "") or ""
    ).strip()
    decode_extra = (
        state.get("last_restart_pd_decode_extra_args") or os.environ.get("PD_DECODE_EXTRA_ARGS", "") or ""
    ).strip()

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

    out.update(
        {
            "pd_prefill_nodes": pn,
            "pd_decode_nodes": dn,
            "pd_prefill_tp": ptp,
            "pd_decode_tp": dtp,
            "pd_transfer_backend": tb,
            "pd_ib_device": ib,
            "pd_prefill_ep": pep,
            "pd_decode_ep": dep,
            "pd_prefill_extra_args": prefill_extra,
            "pd_decode_extra_args": decode_extra,
        }
    )
    return out


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
    try:
        tp_int = int(tp if tp is not None else (state.get("last_restart_tp") or os.environ.get("TP", "") or 0))
    except (TypeError, ValueError):
        tp_int = 0
    try:
        ep_int = int(ep if ep is not None else (state.get("last_restart_ep") or os.environ.get("EP", "") or 1))
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
    pd_mode: str | None = None,
    pd_prefill_nodes: int | None = None,
    pd_decode_nodes: int | None = None,
    pd_prefill_tp: int | None = None,
    pd_decode_tp: int | None = None,
    pd_transfer_backend: str | None = None,
    pd_ib_device: str | None = None,
    health_timeout_s: int = DEFAULT_HEALTH_TIMEOUT_S,
    poll_interval_s: int = 6,
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

    fw, mdl, tp_int, ep_int = _resolve_round_args(framework, model_path, tp, ep)
    pd = _resolve_pd_args(
        pd_mode,
        pd_prefill_nodes,
        pd_decode_nodes,
        pd_prefill_tp,
        pd_decode_tp,
        pd_transfer_backend,
        pd_ib_device,
        tp_int=tp_int,
    )

    # PD-disaggregated x EP cross-check: ep must not exceed either group's TP.
    if pd["pd_mode"] == "disaggregated" and ep_int > 1:
        min_grp_tp = min(pd["pd_prefill_tp"], pd["pd_decode_tp"])
        if ep_int > min_grp_tp:
            raise ServerRestartFailed(
                f"ep={ep_int} > min(pd_prefill_tp={pd['pd_prefill_tp']}, "
                f"pd_decode_tp={pd['pd_decode_tp']})={min_grp_tp}; "
                "lower --ep or raise the smaller per-group TP."
            )

    # Apply Magpie's sglang DEFAULT_ARGS only for sglang (vllm has its own).
    if fw == "sglang":
        extra_server_args = _merge_sglang_defaults(extra_server_args)

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

    async with _get_restart_lock():
        saved_trace_env = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR")
        if torch_profiler_dir:
            try:
                Path(torch_profiler_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ServerRestartFailed(f"cannot mkdir torch_profiler_dir {torch_profiler_dir!r}: {exc}") from exc
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = torch_profiler_dir
        else:
            # No profiler this round — drop stale env so the launcher doesn't reuse a previous round's path.
            os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)

        # Per-variant env overrides → forwarded to the SSH-launched sglang via
        # ``multi_node/commands/infera.py::_collect_forward_env`` (reads this control env).
        saved_fwd_env = os.environ.get("HYPERLOOM_MN_EXTRA_FWD_ENV")
        saved_unset_fwd_env = os.environ.get("HYPERLOOM_MN_UNSET_FWD_ENV")
        unset_keys = [str(k).strip() for k in (unset_env or []) if str(k).strip()]
        if extra_env:
            safe_env = filter_forward_env({str(k): str(v) for k, v in extra_env.items()}, warn_on_drop=True)
            os.environ["HYPERLOOM_MN_EXTRA_FWD_ENV"] = json.dumps(safe_env)
        else:
            os.environ.pop("HYPERLOOM_MN_EXTRA_FWD_ENV", None)
        if unset_keys:
            os.environ["HYPERLOOM_MN_UNSET_FWD_ENV"] = json.dumps(unset_keys)
        else:
            os.environ.pop("HYPERLOOM_MN_UNSET_FWD_ENV", None)

        # Multi-node TraceLens SGLang patch fan-out (fail-soft).
        try:
            from ._server_patcher import _tracelens_patch_enabled
        except Exception:  # noqa: BLE001
            _tracelens_patch_enabled_fn = lambda: True  # noqa: E731 - safe default
        else:
            _tracelens_patch_enabled_fn = _tracelens_patch_enabled
        if _tracelens_patch_enabled_fn() and (os.environ.get("TRACELENS_ROOT", "").strip()):
            try:
                from hyperloom.inference_optimizer.multi_node.cli import cmd_apply_tracelens_patch

                patch_ns = argparse.Namespace(
                    tracelens_root=os.environ.get("TRACELENS_ROOT", "").strip(),
                    sglang_version_pin=os.environ.get(
                        "HYPERLOOM_SGLANG_VERSION_PIN",
                        "",
                    ).strip()
                    or None,
                    print_logs=False,
                    poll_interval=poll_interval_s,
                    poll_timeout=int(
                        os.environ.get(
                            "HYPERLOOM_MN_POLL_TIMEOUT_S",
                            str(health_timeout_s),
                        )
                        or health_timeout_s
                    ),
                )
                patch_rc = await asyncio.to_thread(cmd_apply_tracelens_patch, patch_ns)
                if patch_rc != 0:
                    log.warning(
                        "restart_server_for_round: TraceLens SGLang patch fan-out "
                        "returned rc=%d; proceeding with restart (trace will be "
                        "unannotated; tracelens splitter may report "
                        "trace_split_no_steady_state until patches succeed)",
                        patch_rc,
                    )
            except Exception as exc:  # noqa: BLE001 - fail-soft envelope
                log.warning(
                    "restart_server_for_round: TraceLens patch fan-out raised (%s); proceeding with restart (fail-soft)",
                    exc,
                )

        try:
            # Local import to keep httpx out of the single-node import path.
            from hyperloom.inference_optimizer.multi_node.cli import cmd_restart_server, _resolve_poll_timeout_s

            poll_timeout_s = int(
                os.environ.get(
                    "HYPERLOOM_MN_POLL_TIMEOUT_S",
                    str(health_timeout_s),
                )
                or health_timeout_s
            )
            health_wait_s = int(
                os.environ.get(
                    "HYPERLOOM_MN_HEALTH_WAIT_S",
                    str(health_timeout_s),
                )
                or health_timeout_s
            )
            # Align launch-driver poll with /health wait for JIT-heavy MoE runs.
            poll_timeout_s = max(poll_timeout_s, _resolve_poll_timeout_s())

            # aiter kernels JIT-compile + autotune on first use (server log: "not found tuned config in
            # /tmp/aiter_configs"); a cold compile can exceed the default 900s gate and false-fail an
            # otherwise-healthy variant (the doomed attempt then burns a reclaim+retry cycle before the now-warm
            # relaunch succeeds).
            if "HYPERLOOM_MN_HEALTH_WAIT_S" not in os.environ and _uses_aiter(extra_server_args, pd, extra_env):
                _aiter_wait = int(os.environ.get("HYPERLOOM_MN_HEALTH_WAIT_AITER_S", "1800") or 1800)
                if _aiter_wait > health_wait_s:
                    log.info(
                        "restart_server_for_round: aiter kernels detected; widening "
                        "worker /health wait %ds -> %ds for cold JIT/autotune "
                        "(HYPERLOOM_MN_HEALTH_WAIT_AITER_S)",
                        health_wait_s,
                        _aiter_wait,
                    )
                    health_wait_s = _aiter_wait
                    poll_timeout_s = max(poll_timeout_s, _aiter_wait)

            ns = argparse.Namespace(
                framework=fw,
                model=mdl,
                tp=tp_int,
                ep=ep_int,
                extra_args=extra_server_args or "",
                pid_file=None,
                log_file=None,
                no_wait_health=False,
                print_logs=False,
                poll_interval=poll_interval_s,
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
                prev_resume = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING")
                if force_full:
                    os.environ["MULTI_NODE_RESTART_RESUME_RUNNING"] = "0"
                try:
                    rc = await asyncio.to_thread(cmd_restart_server, ns)
                except Exception as exc:  # noqa: BLE001
                    raise ServerRestartFailed(f"cmd_restart_server raised: {exc!r}") from exc
                finally:
                    if force_full:
                        if prev_resume is None:
                            os.environ.pop("MULTI_NODE_RESTART_RESUME_RUNNING", None)
                        else:
                            os.environ["MULTI_NODE_RESTART_RESUME_RUNNING"] = prev_resume

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
                    await _wait_for_workers_ready_async(
                        timeout_s=health_wait_s,
                        poll_every_s=int(os.environ.get("HYPERLOOM_MN_HEALTH_POLL_S", "10")),
                    )
                    await _wait_for_server_health_async(
                        timeout_s=health_wait_s,
                        poll_every_s=int(os.environ.get("HYPERLOOM_MN_HEALTH_POLL_S", "10")),
                    )
                    # The reachable /health above can be a pod-pinned head address; the benchmark dials the published
                    # ClusterIP Service, whose endpoints lag readiness after a restart.
                    await _wait_for_published_service_ready_async(
                        timeout_s=_published_ready_timeout_s(),
                        poll_every_s=int(os.environ.get("HYPERLOOM_MN_HEALTH_POLL_S", "10")),
                    )
                except ServerRestartFailed as exc:
                    _collect_worker_server_logs(_read_state() or {}, str(exc))
                    raise
                except Exception as exc:  # noqa: BLE001
                    _collect_worker_server_logs(_read_state() or {}, repr(exc))
                    raise ServerRestartFailed(f"post-launch /health wait raised: {exc!r}") from exc

            try:
                await _restart_and_wait(force_full_restart)
            except ServerRestartFailed as first_exc:
                # C — multi-node VRAM reclaim before exactly one retry.
                if os.environ.get(_MN_RESTART_RECLAIM_RETRY_ENV, "1").strip().lower() in {
                    "0",
                    "false",
                    "no",
                    "off",
                }:
                    raise
                log.warning(
                    "restart_server_for_round: restart failed (%s); attempting "
                    "best-effort remote kill-inference + one retry",
                    first_exc,
                )
                try:
                    from hyperloom.inference_optimizer.multi_node.cli import kill_inference_for_kernel_agent_best_effort

                    await asyncio.to_thread(kill_inference_for_kernel_agent_best_effort)
                except Exception as reclaim_exc:  # noqa: BLE001 - reclaim is best-effort
                    log.warning(
                        "restart_server_for_round: remote kill-inference reclaim raised (%s); retrying restart anyway",
                        reclaim_exc,
                    )
                try:
                    await _restart_and_wait(force_full=True)
                except ServerRestartFailed as retry_exc:
                    raise retry_exc from first_exc
        finally:
            # Restore env so this round's profiler path doesn't leak forward.
            if saved_trace_env is None:
                os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)
            else:
                os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = saved_trace_env
            # Symmetric restore for the per-variant env forwarding control var.
            if saved_fwd_env is None:
                os.environ.pop("HYPERLOOM_MN_EXTRA_FWD_ENV", None)
            else:
                os.environ["HYPERLOOM_MN_EXTRA_FWD_ENV"] = saved_fwd_env
            if saved_unset_fwd_env is None:
                os.environ.pop("HYPERLOOM_MN_UNSET_FWD_ENV", None)
            else:
                os.environ["HYPERLOOM_MN_UNSET_FWD_ENV"] = saved_unset_fwd_env


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
        except Exception as exc:  # noqa: BLE001 — fail-soft
            log.warning(
                "infera profile %s -> %s failed (%s); continuing",
                route,
                url,
                exc,
            )


def _probe_generated_tokens(data: object) -> int:
    """Best-effort count of tokens a /v1/completions probe actually generated."""
    from hyperloom.inference_optimizer.multi_node._internal.serving_probe import generated_tokens

    return generated_tokens(data)


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


def _worker_detokenizer_wedged(shared_dir: str, ip: str) -> bool:
    """Best-effort: True when worker ``ip``'s server log shows a persistent detokenizer wedge -- weights are loaded but the detokenizer never heartbeats, so the engine's /health never flips and the HTTP port never binds."""
    import re as _re

    if not shared_dir:
        return False
    log_path = Path(shared_dir) / ("mn_infera_server_" + str(ip) + "_r0.log")
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    # >=15 recent markers ~= 30s+ of continuous stall (marker ~every 2s).
    return len(_re.findall(r"Health check failed.*detokenizer", tail, _re.IGNORECASE)) >= 15


def _worker_startup_crashed(shared_dir: str, ip: str) -> str | None:
    """Best-effort: return a short reason when worker ``ip``'s server log shows a NON-RECOVERABLE startup failure (so /health will never flip), else None."""
    import re as _re

    if not shared_dir:
        return None
    log_path = Path(shared_dir) / ("mn_infera_server_" + str(ip) + "_r0.log")
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
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


async def _wait_for_workers_ready_async(timeout_s: int, poll_every_s: int = 10) -> None:
    """Wait for every prefill/decode worker's own /health to return 200 before the frontend serving probe runs."""
    import time as _t

    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover
        return
    state = _read_state() or {}
    port = int(os.environ.get("HYPERLOOM_MN_WORKER_PORT", "30000") or 30000)
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
    if not targets:
        return
    started = _t.monotonic()
    ready: set[str] = set()
    # Detokenizer-wedge fast-fail: bail early on a wedged worker instead of burning the full timeout.
    _wedge_grace = int(os.environ.get("HYPERLOOM_MN_WORKER_WEDGE_GRACE_S", "420") or 420)
    _wedge_enabled = os.environ.get("HYPERLOOM_MN_WORKER_WEDGE_FASTFAIL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    # Startup-crash fast-fail: an argparse rejection / fatal exit is terminal, so bail within seconds instead of the
    # full gate.
    _crash_grace = int(os.environ.get("HYPERLOOM_MN_WORKER_CRASH_GRACE_S", "45") or 45)
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
                except Exception:  # noqa: BLE001
                    pass
            if len(ready) == len(targets):
                log.info(
                    "all %d worker(s) /health-ready after %ds; proceeding to frontend probe", len(targets), elapsed
                )
                return
            if elapsed > timeout_s:
                not_ready = [ip for _r, ip, _p in targets if ip not in ready]
                raise ServerRestartFailed(
                    "workers not /health-ready within "
                    + str(timeout_s)
                    + "s: "
                    + str(not_ready)
                    + " (a PD leg is still initializing, e.g. prefill "
                    "mooncake transfer engine)"
                )
            if _wedge_enabled and elapsed > _crash_grace:
                _shared_c = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
                if _shared_c.startswith("/") and "$" not in _shared_c:
                    for _r, ip, _p in targets:
                        if ip in ready:
                            continue
                        _reason = _worker_startup_crashed(_shared_c, ip)
                        if _reason:
                            raise ServerRestartFailed(
                                "worker " + ip + " server crashed on startup "
                                "(non-recoverable; /health will never come up): "
                                + _reason
                                + " -- fast-failed after "
                                + str(elapsed)
                                + "s instead of the full "
                                + str(timeout_s)
                                + "s gate"
                            )
            if _wedge_enabled and elapsed > _wedge_grace:
                _shared = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
                if _shared.startswith("/") and "$" not in _shared:
                    wedged = [
                        ip for _r, ip, _p in targets if ip not in ready and _worker_detokenizer_wedged(_shared, ip)
                    ]
                    if wedged:
                        raise ServerRestartFailed(
                            "worker(s) detokenizer-wedged (weights loaded but "
                            "detokenizer not heartbeating; /health never up): "
                            + str(wedged)
                            + " after "
                            + str(elapsed)
                            + "s; fast-failed "
                            "instead of waiting the full " + str(timeout_s) + "s gate"
                        )
            await asyncio.sleep(poll_every_s)


def _shared_worker_log_tail(ip: str, max_bytes: int = 2_000_000) -> str | None:
    """Best-effort tail of a worker's shared-FS sglang server log."""
    import glob as _glob

    shared = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
    if not shared.startswith("/") or "$" in shared:
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


def _collect_worker_server_logs(state: dict, reason: str) -> None:
    """Best-effort: capture each prefill/decode pod sglang server log into ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/server_logs`` when a restart fails its health/serving probe, so a 503 / KV handoff / cold-JIT failure leaves a post-mortem after the pods are torn down."""
    import time as _t

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
    except Exception:
        return
    key_path = state.get("ssh_key_path")
    known_hosts = state.get("ssh_known_hosts")
    remote = os.environ.get("HYPERLOOM_MN_SERVER_LOG_PATH", str(Path(tempfile.gettempdir()) / "mn_infera_server.log"))
    default_port = int(state.get("ssh_port") or 2233)
    ssh_run = None
    if key_path and known_hosts:
        try:
            from hyperloom.inference_optimizer.multi_node._internal.ssh_client import ssh_run as _ssh_run

            ssh_run = _ssh_run
        except Exception:
            ssh_run = None
    for role in ("prefill", "decode", "worker"):
        for pod in state.get(role + "_pods") or []:
            ip = pod.get("podIP")
            if not ip:
                continue
            tag = pod.get("podId") or ip
            body = ""
            source = ""
            # 1) Shared-FS read (preferred: real trace, no SSH, survives teardown).
            shared_tail = _shared_worker_log_tail(str(ip))
            if shared_tail is not None:
                body = shared_tail
                source = "shared-fs"
            elif ssh_run is not None:
                # 2) Fallback: SSH tail of the pod-local log.
                try:
                    port = int(pod.get("sshPort") or default_port)
                    cp = ssh_run(
                        ip,
                        "tail -c 2000000 " + remote + " 2>/dev/null",
                        key_path=key_path,
                        known_hosts=known_hosts,
                        port=port,
                        timeout=60,
                    )
                    body = cp.stdout or ""
                    if cp.stderr:
                        body += "\n--- ssh stderr ---\n" + cp.stderr
                    source = "ssh:" + str(remote)
                except Exception:
                    continue
            else:
                continue
            try:
                dest = os.path.join(out_dir, ts + "_" + role + "_" + str(tag) + ".log")
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write("# collected on restart failure (" + source + "): " + str(reason) + "\n")
                    fh.write(body)
                log.info("collected worker server log -> %s", dest)
            except Exception:
                continue


async def _wait_for_server_health_async(
    timeout_s: int = 1800,
    poll_every_s: int = 10,
) -> None:
    """Poll the multi_node service_url /health until 200 or timeout."""
    import time as _time

    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover
        log.warning("httpx not available; skipping post-restart /health wait")
        return

    from hyperloom.inference_optimizer.multi_node._internal.external_state import reachable_service_url

    state = _read_state() or {}
    service_url = reachable_service_url(state)
    backend = str(state.get("backend") or "").strip().lower()

    if not service_url:
        log.warning("no service_url in state; skipping post-restart /health wait")
        return

    health_url = service_url.rstrip("/") + "/health"
    started = _time.monotonic()
    last_err = ""
    log.info(
        "post-restart /health wait: url=%s timeout_s=%d poll_every_s=%d",
        health_url,
        timeout_s,
        poll_every_s,
    )
    # Infera-ONLY serving-readiness extension.
    wait_model_ready = backend == "infera"
    # When wait_model_ready is on, also do a 1-token completion probe to confirm workers actually serve traffic
    # (Infera registers models in /v1/models before the worker is ready to accept requests; this causes the first
    # benchmark to get 503 "Model temporarily unavailable" for every request, surfacing as `completed=0` →
    # `baseline_failed`).
    models_url = service_url.rstrip("/") + "/v1/models"
    completions_url = service_url.rstrip("/") + "/v1/completions"
    health_ok_at = None
    models_ready_at = None
    consecutive_completion_ok = 0
    # require N consecutive successful completions before declaring ready
    completion_probe_required = int(os.environ.get("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", "2") or 2)
    # Probe with >1 token + ignore_eos so the request must traverse the decode leg (prefill alone can serve the first
    # token): a PD run with an unready or KV-broken decode then fails the probe instead of passing on a prefill-only
    # 200.
    completion_probe_tokens = int(os.environ.get("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", "8") or 8)
    completion_probe_min_tokens = int(os.environ.get("HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS", "2") or 2)
    # Fast-fail grace: if /health is up but /v1/models never registers within this many seconds, the GPU workers
    # crashed on launch (e.g. an unrecognized server flag) rather than slowly loading — bail early instead of burning
    # the full timeout_s.
    models_empty_grace_s = int(os.environ.get("HYPERLOOM_MN_MODELS_EMPTY_GRACE_S", "600") or 600)
    # Fast-fail when /v1/models is populated but /v1/completions never serves within this window (PD prefill<->decode
    # KV handoff broken on restart -> HTTP 503 every probe).
    completions_grace_s = int(os.environ.get("HYPERLOOM_MN_COMPLETIONS_GRACE_S", "480") or 480)
    async with _httpx.AsyncClient(timeout=15.0) as client:
        while True:
            elapsed = int(_time.monotonic() - started)
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
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
                    # /v1/models probe
                    try:
                        mresp = await client.get(models_url)
                        if mresp.status_code == 200:
                            try:
                                data = mresp.json()
                            except Exception:
                                data = {}
                            models = data.get("data") if isinstance(data, dict) else None
                            if isinstance(models, list) and len(models) > 0:
                                if models_ready_at is None:
                                    models_ready_at = elapsed
                                    log.info(
                                        "post-restart /v1/models populated after %ds (n=%d, health_ok_at=%ds); now probing /v1/completions",
                                        elapsed,
                                        len(models),
                                        health_ok_at,
                                    )
                                # Worker-readiness probe: tiny completion.
                                model_id = ""
                                try:
                                    model_id = str(models[0].get("id") or "") if isinstance(models[0], dict) else ""
                                except Exception:
                                    model_id = ""
                                if not model_id:
                                    last_err = "completion_probe: no model id"
                                    consecutive_completion_ok = 0
                                else:
                                    try:
                                        cresp = await client.post(
                                            completions_url,
                                            json={
                                                "model": model_id,
                                                "prompt": "hi",
                                                "max_tokens": completion_probe_tokens,
                                                "temperature": 0,
                                                "ignore_eos": True,
                                                "stream": False,
                                            },
                                        )
                                        if cresp.status_code == 200:
                                            try:
                                                gen_toks = _probe_generated_tokens(cresp.json())
                                            except Exception:
                                                gen_toks = 0
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
                                                # HTTP 200 but decode leg produced nothing (unready / broken PD KV
                                                # handoff) — do NOT count as ready.
                                                consecutive_completion_ok = 0
                                                last_err = (
                                                    f"completion_probe_zero_tokens gen={gen_toks} "
                                                    f"(need>={completion_probe_min_tokens}; decode leg not serving)"
                                                )
                                        else:
                                            consecutive_completion_ok = 0
                                            last_err = f"completion_probe_http={cresp.status_code}"
                                    except Exception as cexc:  # noqa: BLE001
                                        consecutive_completion_ok = 0
                                        last_err = f"completion_probe {format_exc_brief(cexc, limit=80)}"
                            else:
                                last_err = f"models_empty (health_ok_at={health_ok_at}s)"
                                consecutive_completion_ok = 0
                        else:
                            last_err = f"models_http_status={mresp.status_code}"
                    except Exception as mexc:  # noqa: BLE001
                        last_err = f"models_probe {format_exc_brief(mexc, limit=80)}"
                else:
                    last_err = f"http_status={resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_err = format_exc_brief(exc, limit=120)
            if _models_empty_too_long(
                elapsed=elapsed,
                health_ok_at=health_ok_at,
                models_ready_at=models_ready_at,
                grace_s=models_empty_grace_s,
            ):
                raise ServerRestartFailed(
                    f"/v1/models still empty {elapsed - (health_ok_at or 0)}s after /health "
                    f"(> grace {models_empty_grace_s}s); GPU workers crashed on launch "
                    f"(url={models_url}, last_err={last_err})"
                )
            if (
                completions_grace_s > 0
                and models_ready_at is not None
                and (elapsed - models_ready_at) > completions_grace_s
            ):
                raise ServerRestartFailed(
                    f"/v1/models populated at {models_ready_at}s but /v1/completions still "
                    f"not serving {elapsed - models_ready_at}s later (> grace {completions_grace_s}s); "
                    f"serving broken, likely PD KV handoff on restart "
                    f"(url={completions_url}, last_err={last_err})"
                )
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    f"server /health did not return 200 within {timeout_s}s (url={health_url}, last_err={last_err})"
                )
            if elapsed % 60 < poll_every_s:
                log.info(
                    "post-restart /health still waiting t=%ds last_err=%s",
                    elapsed,
                    last_err,
                )
            await asyncio.sleep(poll_every_s)


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


def _published_ready_timeout_s() -> int:
    """Resolve the published-service gate budget from the environment."""
    try:
        return int(os.environ.get("HYPERLOOM_MN_PUBLISHED_READY_S", "300"))
    except ValueError:
        return 300


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive int knob from the environment, clamped and junk-tolerant."""
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


async def _wait_for_published_service_ready_async(
    timeout_s: int = 300,
    poll_every_s: int = 10,
) -> None:
    """Gate on the published ``service_url`` the benchmark dials, not only the reachable address :func:`_wait_for_server_health_async` proves."""
    import time as _time

    if timeout_s <= 0:
        log.info("post-restart published-service gate disabled (HYPERLOOM_MN_PUBLISHED_READY_S<=0)")
        return

    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover - httpx is backfilled by preflight
        return

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
    required_ok = _env_int("HYPERLOOM_MN_PUBLISHED_READY_OK_STREAK", 2)
    ok_streak = 0
    # Skip only after this many CONSECUTIVE name-resolution failures, so a CoreDNS blip inside the readiness window
    # (which resolves on a later poll) does not disable the gate exactly when it is needed.
    dns_skip_after = _env_int("HYPERLOOM_MN_PUBLISHED_DNS_SKIP_AFTER", 6)
    dns_failures = 0
    started = _time.monotonic()
    last_err = ""
    log.info(
        "post-restart published-service /health wait: url=%s timeout_s=%d",
        health_url,
        timeout_s,
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
            except Exception as exc:  # noqa: BLE001
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
                            dns_failures * poll_every_s,
                        )
                        return
                    ok_streak = 0
                    last_err = f"name resolution failed ({dns_failures}/{dns_skip_after})"
                else:
                    dns_failures = 0  # reached the host: a real refuse/5xx, keep waiting
                    ok_streak = 0
                    last_err = format_exc_brief(exc, limit=120)
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    f"published service /health did not return 200 within {timeout_s}s "
                    f"(url={health_url}, last_err={last_err}); the endpoint the benchmark "
                    f"dials had no ready backend (post-restart readiness window did not close)"
                )
            await asyncio.sleep(poll_every_s)


__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_S",
    "ServerRestartFailed",
    "_merge_sglang_defaults",
    "restart_server_for_round",
]
