# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node-only: per-round sglang/vllm restart helper.

Single-node Magpie restarts the server on every benchmark invocation, baking
that round's flags + profiler env into a fresh process. Multi-node used to run
the whole grid against one long-lived server, silently dropping later variants'
flags. This helper closes the gap: every executor calls
:func:`restart_server_for_round` before spawning Magpie, invoking
``multi_node restart-server`` with the round's framework/model/tp + extra-args
(and a per-round profiler trace dir).

No-op in single-node mode (``is_multi_node()`` False). Fail-fast: any failure
raises :class:`ServerRestartFailed`, which callers let bubble so the round is
marked failed rather than benchmarking a stale/half-dead server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from ._multi_node_env import _read_state, is_multi_node


log = logging.getLogger(__name__)


# Default /health poll timeout. Tightened from 1800s to 900s (~2x normal MoE
# cold-start headroom) so an incompatible-config variant aborts ~2x faster;
# override per-run via HYPERLOOM_MN_HEALTH_WAIT_S.
DEFAULT_HEALTH_TIMEOUT_S = 900  # 15 min.

# Magpie's sglang_mi*x.sh DEFAULT_ARGS, re-applied in multi-node so tput numbers
# stay comparable to single-node. We diverge on --mem-fraction-static (0.75 vs
# single-node 0.8) because cross-node RDMA buffers eat headroom (0.8 OOMs on
# DSr1 671B FP8 when the second node joins). ``_merge_sglang_defaults`` skips a
# default when the user already set ``flag_name``.
_SGLANG_DEFAULT_TOKENS: tuple[tuple[str, str], ...] = (
    ("--mem-fraction-static", "--mem-fraction-static=0.75"),
    ("--disable-radix-cache", "--disable-radix-cache"),
)


def _merge_sglang_defaults(extra_args: str) -> str:
    """Append Magpie's DEFAULT_ARGS that the user did not already set.

    Mirrors ``sglang_mi300x.sh:74-79``; a caller's explicit value for a flag
    wins and that default is dropped.
    """
    user = (extra_args or "").strip()
    parts = [user] if user else []
    for flag_name, default_token in _SGLANG_DEFAULT_TOKENS:
        if flag_name in user:
            continue
        parts.append(default_token)
    return " ".join(p for p in parts if p)

# Round-trip context so callers (profile.py) can recover the trace dir the
# server was restarted with, even after this helper restored the env.
_LAST_ROUND_TRACE_DIR: str = ""


class ServerRestartFailed(RuntimeError):
    """Raised when the per-round multi-node server restart did not succeed."""


def last_round_trace_dir() -> str:
    """Return the trace dir the most recent restart was wired with (or '').

    Returns:
        str: The most recent round's profiler trace dir, or ``""`` if none.
    """
    return _LAST_ROUND_TRACE_DIR


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
    """Resolve PD knobs with state.json + env fallback.

    Returns a flat dict of resolved PD values for the multi_node CLI Namespace.
    Resolution per field: explicit kwarg > ``state["last_restart_pd_*"]`` >
    ``$PD_*`` env > defaults (mode=colocated, prefill/decode TP=tp). Validates
    ``pd_mode`` and disaggregated prefill/decode TP.
    """
    state = _read_state()
    mode = (
        pd_mode
        or state.get("last_restart_pd_mode")
        or os.environ.get("PD_MODE", "")
        or "colocated"
    ).strip().lower()
    if mode not in ("colocated", "disaggregated"):
        raise ServerRestartFailed(
            f"unsupported pd_mode {mode!r}; expected 'colocated' or 'disaggregated'"
        )

    out: dict = {"pd_mode": mode}
    if mode == "colocated":
        return out

    # PD disaggregation requires >=2 nodes; defend against a mangled state.json
    # so we surface a recoverable round failure, not a confusing launcher error.
    state_nodes = int(state.get("nodes") or 0)
    if state_nodes < 2:
        raise ServerRestartFailed(
            f"pd_mode=disaggregated requires nodes>=2 but state.json "
            f"reports nodes={state_nodes}. Re-provision the RayJob "
            "with `--nodes >=2` or drop --pd-mode."
        )

    def _intf(kw, sk, ek):
        """Resolve an int field from kwarg > state key > env var.

        Args:
            kw: The explicit kwarg value (wins when not ``None``).
            sk (str): The ``state.json`` key to read next.
            ek (str): The environment variable name to read last.

        Returns:
            int: The first parseable integer found, or ``0`` when none.
        """
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
    ptp = _intf(pd_prefill_tp, "last_restart_pd_prefill_tp", "PD_PREFILL_TP") or tp_int
    dtp = _intf(pd_decode_tp, "last_restart_pd_decode_tp", "PD_DECODE_TP") or tp_int
    tb = (
        pd_transfer_backend
        or state.get("last_restart_pd_transfer_backend")
        or os.environ.get("PD_TRANSFER_BACKEND", "")
        or ""
    ).strip()
    ib = (
        pd_ib_device
        or state.get("last_restart_pd_ib_device")
        or os.environ.get("PD_IB_DEVICE", "")
        or ""
    ).strip()

    state_nodes = int(state.get("nodes") or 0)
    if pn <= 0 or dn <= 0:
        raise ServerRestartFailed(
            f"pd_mode=disaggregated requires pd_prefill_nodes>0 and "
            f"pd_decode_nodes>0; got pn={pn} dn={dn}"
        )
    if state_nodes > 0 and pn + dn != state_nodes:
        raise ServerRestartFailed(
            f"pd_prefill_nodes ({pn}) + pd_decode_nodes ({dn}) must equal "
            f"total nodes ({state_nodes})"
        )
    if ptp <= 0 or dtp <= 0:
        raise ServerRestartFailed(
            f"pd_prefill_tp ({ptp}) and pd_decode_tp ({dtp}) must be positive"
        )

    out.update({
        "pd_prefill_nodes": pn,
        "pd_decode_nodes": dn,
        "pd_prefill_tp": ptp,
        "pd_decode_tp": dtp,
        "pd_transfer_backend": tb,
        "pd_ib_device": ib,
    })
    return out


def _resolve_round_args(
    framework: str | None,
    model_path: str | None,
    tp: int | None,
    ep: int | None = None,
) -> tuple[str, str, int, int]:
    """Resolve (framework, model, tp, ep) for the restart, with state fallback.

    Resolution per field: explicit kwarg > ``state["last_restart_*"]`` >
    ``$FRAMEWORK`` / ``$MODEL_PATH`` / ``$TP`` / ``$EP`` env > defaults (ep=1).
    Raises :class:`ServerRestartFailed` if model/tp are empty, framework is
    unsupported, or ``ep > tp``.
    """
    state = _read_state()
    fw = (framework or state.get("last_restart_framework")
          or os.environ.get("FRAMEWORK", "sglang") or "sglang").strip().lower()
    mdl = (model_path or state.get("last_restart_model")
           or os.environ.get("MODEL_PATH", "") or "").strip()
    try:
        tp_int = int(tp if tp is not None else (
            state.get("last_restart_tp") or os.environ.get("TP", "") or 0
        ))
    except (TypeError, ValueError):
        tp_int = 0
    try:
        ep_int = int(ep if ep is not None else (
            state.get("last_restart_ep") or os.environ.get("EP", "") or 1
        ))
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
        raise ServerRestartFailed(
            f"unsupported framework {fw!r}; expected 'sglang' or 'vllm'"
        )
    if ep_int > tp_int:
        raise ServerRestartFailed(
            f"ep={ep_int} > tp={tp_int} is not supported by sglang/vllm "
            "(cannot place more expert shards than ranks). Lower --ep or "
            "raise --tp."
        )
    return fw, mdl, tp_int, ep_int


async def restart_server_for_round(
    *,
    extra_server_args: str = "",
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
    """Restart the multi-node inference server for the next Magpie round.

    No-op when ``is_multi_node()`` is False. For multi-node: resolves
    framework/model/tp, mkdirs + exports ``torch_profiler_dir`` via
    ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` (restored afterward), and invokes
    ``cmd_restart_server`` in a thread.

    ``force_full_restart``: scopes ``MULTI_NODE_RESTART_RESUME_RUNNING=0`` for
    this invocation so a fresh kill+launch runs — required after kernel-agent
    fans patched source so sglang re-imports the new modules.

    Raises :class:`ServerRestartFailed` on any failure (callers let it bubble).
    """
    global _LAST_ROUND_TRACE_DIR

    if not is_multi_node():
        return

    fw, mdl, tp_int, ep_int = _resolve_round_args(framework, model_path, tp, ep)
    pd = _resolve_pd_args(
        pd_mode, pd_prefill_nodes, pd_decode_nodes,
        pd_prefill_tp, pd_decode_tp, pd_transfer_backend, pd_ib_device,
        tp_int=tp_int,
    )

    # PD-disaggregated × EP cross-check: ep must not exceed either group's TP.
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

    saved_trace_env = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR")
    if torch_profiler_dir:
        try:
            Path(torch_profiler_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ServerRestartFailed(
                f"cannot mkdir torch_profiler_dir {torch_profiler_dir!r}: {exc}"
            ) from exc
        os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = torch_profiler_dir
        _LAST_ROUND_TRACE_DIR = torch_profiler_dir
    else:
        # No profiler this round — drop stale env so the launcher doesn't
        # reuse a previous round's path.
        os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)
        _LAST_ROUND_TRACE_DIR = ""

    # Multi-node TraceLens SGLang patch fan-out (fail-soft). The controller
    # can't ``import sglang`` (it lives in the pods), so the local patcher
    # skips; without these patches the trace splitter ends every profile round
    # in ``trace_split_no_steady_state``. Fan out (idempotent per pod) before
    # ``cmd_restart_server``; on failure log a warning and proceed (trace
    # unannotated, but other phases keep working).
    try:
        from ._server_patcher import _tracelens_patch_enabled
    except Exception:  # noqa: BLE001
        _tracelens_patch_enabled_fn = lambda: True  # noqa: E731 - safe default
    else:
        _tracelens_patch_enabled_fn = _tracelens_patch_enabled
    if _tracelens_patch_enabled_fn() and (
        os.environ.get("TRACELENS_ROOT", "").strip()
    ):
        try:
            from ...multi_node.cli import cmd_apply_tracelens_patch

            patch_ns = argparse.Namespace(
                tracelens_root=os.environ.get("TRACELENS_ROOT", "").strip(),
                sglang_version_pin=os.environ.get(
                    "HYPERLOOM_SGLANG_VERSION_PIN", "",
                ).strip() or None,
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
                "restart_server_for_round: TraceLens patch fan-out raised "
                "(%s); proceeding with restart (fail-soft)",
                exc,
            )

    try:
        # Local import to keep httpx out of the single-node import path.
        from ...multi_node.cli import cmd_restart_server, _resolve_poll_timeout_s

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
            # PD knobs; colocated mode passes only pd_mode.
            pd_mode=pd.get("pd_mode", "colocated"),
            pd_prefill_nodes=pd.get("pd_prefill_nodes", 0),
            pd_decode_nodes=pd.get("pd_decode_nodes", 0),
            pd_prefill_tp=pd.get("pd_prefill_tp", 0),
            pd_decode_tp=pd.get("pd_decode_tp", 0),
            pd_transfer_backend=pd.get("pd_transfer_backend", ""),
            pd_ib_device=pd.get("pd_ib_device", ""),
            pd_bootstrap_port=8998,
            pd_vllm_router_cmd="",
        )
        from ._multi_node_env import log_mn_banner
        log_mn_banner(
            "server_restart", log,
            framework=fw, tp=tp_int, ep=ep_int,
            pd_mode=pd.get("pd_mode"),
            trace_dir=torch_profiler_dir or "",
        )
        log.info(
            "restart_server_for_round: framework=%s tp=%d ep=%d pd_mode=%s "
            "pd_prefill=%dx tp%d pd_decode=%dx tp%d backend=%r ib=%r "
            "extra_args=%r torch_profiler_dir=%r",
            fw, tp_int, ep_int, pd.get("pd_mode"),
            pd.get("pd_prefill_nodes", 0), pd.get("pd_prefill_tp", 0),
            pd.get("pd_decode_nodes", 0), pd.get("pd_decode_tp", 0),
            pd.get("pd_transfer_backend", ""), pd.get("pd_ib_device", ""),
            extra_server_args, torch_profiler_dir,
        )

        # After kernel-agent patches sglang source, the resume fast-path would
        # keep the old module imports; scope an override for this invocation so
        # a fresh kill+launch runs.
        prev_resume = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING")
        if force_full_restart:
            os.environ["MULTI_NODE_RESTART_RESUME_RUNNING"] = "0"
        try:
            rc = await asyncio.to_thread(cmd_restart_server, ns)
        except Exception as exc:  # noqa: BLE001
            raise ServerRestartFailed(
                f"cmd_restart_server raised: {exc!r}"
            ) from exc
        finally:
            if force_full_restart:
                if prev_resume is None:
                    os.environ.pop("MULTI_NODE_RESTART_RESUME_RUNNING", None)
                else:
                    os.environ["MULTI_NODE_RESTART_RESUME_RUNNING"] = prev_resume

        if rc != 0:
            raise ServerRestartFailed(
                f"cmd_restart_server returned non-zero rc={rc} "
                f"(framework={fw} tp={tp_int} extra_args={extra_server_args!r})"
            )

        # cmd_restart_server returns when actors are spawned, but a cold MoE
        # weight-load can need 20-30 min before /health flips; poll it here so
        # the downstream baseline doesn't fire against a not-yet-ready server.
        try:
            await _wait_for_server_health_async(
                timeout_s=health_wait_s,
                poll_every_s=int(os.environ.get(
                    "HYPERLOOM_MN_HEALTH_POLL_S", "10",
                )),
            )
        except ServerRestartFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ServerRestartFailed(
                f"post-launch /health wait raised: {exc!r}"
            ) from exc
    finally:
        # Restore env so this round's profiler path doesn't leak forward.
        if saved_trace_env is None:
            os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = saved_trace_env




async def _wait_for_server_health_async(
    timeout_s: int = 1800,
    poll_every_s: int = 10,
) -> None:
    """Poll the multi_node service_url /health until 200 or timeout.

    Reads ``service_url`` from ``state.json``; rewrites a ClusterIP DNS URL to
    ``head_pod_ip`` when available so the sandbox can reach it directly.
    """
    import time as _time
    import re as _re
    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover
        log.warning("httpx not available; skipping post-restart /health wait")
        return

    state = _read_state() or {}
    service_url = str(state.get("service_url") or "").strip()
    head_ip = str(state.get("head_pod_ip") or "").strip()
    if head_ip and ".svc.cluster.local" in service_url:
        m = _re.search(r":(\d+)$", service_url)
        port = m.group(1) if m else "8888"
        service_url = f"http://{head_ip}:{port}"

    if not service_url:
        log.warning("no service_url in state; skipping post-restart /health wait")
        return

    health_url = service_url.rstrip("/") + "/health"
    started = _time.monotonic()
    last_err = ""
    log.info(
        "post-restart /health wait: url=%s timeout_s=%d poll_every_s=%d",
        health_url, timeout_s, poll_every_s,
    )
    async with _httpx.AsyncClient(timeout=5.0) as client:
        while True:
            elapsed = int(_time.monotonic() - started)
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    log.info(
                        "post-restart /health OK after %ds (url=%s)",
                        elapsed, health_url,
                    )
                    return
                last_err = f"http_status={resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    f"server /health did not return 200 within {timeout_s}s "
                    f"(url={health_url}, last_err={last_err})"
                )
            if elapsed % 60 < poll_every_s:
                log.info(
                    "post-restart /health still waiting t=%ds last_err=%s",
                    elapsed, last_err,
                )
            await asyncio.sleep(poll_every_s)


__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_S",
    "ServerRestartFailed",
    "_merge_sglang_defaults",
    "last_round_trace_dir",
    "restart_server_for_round",
]
