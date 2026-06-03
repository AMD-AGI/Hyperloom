"""Multi-node-only: per-round sglang/vllm restart helper.

Why this exists
---------------
Single-node Magpie (`sglang_mi300x.sh PHASE=all`) restarts the inference
server on EVERY benchmark invocation, baking that round's server-side
flags (e.g. ``--quantization fp8 --enable-torch-compile``) and profiling
env (``SGLANG_TORCH_PROFILER_DIR=...``) into the fresh server process.
That keeps every test variant deterministic and isolated.

Multi-node mode used to skip this — Magpie was forced into PHASE=client
so it never relaunched the server, and orchestrator executors ran their
whole grid against ONE long-lived sglang instance booted by the initial
``multi_node create-rayjob``. Server-side flags from later variants were
silently dropped, and the torch profiler env was frozen at whatever
``cli.py`` exported when the RayJob first came up.

This helper closes that gap: every executor that is about to spawn a
Magpie subprocess calls :func:`restart_server_for_round` first; the
helper invokes ``inference_optimizer.multi_node restart-server`` with
the round's framework/model/tp + ``--extra-args`` and (for profile
rounds) a per-round ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` so the next
sglang launch picks them up via the existing
``launch_multinode.py --torch-profiler-dir`` plumb.

Single-node short-circuit
-------------------------
:func:`is_multi_node` returns ``False`` ⇒ this helper is a no-op. The
single-node path keeps using Magpie's own server lifecycle; nothing in
``baseline.py`` / ``profile.py`` / ``_grid_runner.py`` changes
behaviourally for ``--nodes 1``.

Failure semantics (Q4 = fail-fast)
----------------------------------
On any failure (missing model/tp in state, ``cmd_restart_server`` non-
zero return, /health flip timeout) we raise :class:`ServerRestartFailed`.
Callers should let this bubble — the round becomes a ``failed`` task
and the coordinator surfaces a clear error rather than benchmarking
against a stale or half-dead server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from ._multi_node_env import _read_state, is_multi_node


log = logging.getLogger(__name__)


# Default poll timeout for the dashboard-side launch.
# Cold-load of large MoE models (DeepSeek-R1 671B, Llama-405B) can take
# 20-30 min through aiter JIT; warm restarts typically finish in 3-5 min.
# 1800 s (30 min) leaves headroom on cold while still failing-fast on
# truly stuck launches. Override per-call via ``health_timeout_s``.
DEFAULT_HEALTH_TIMEOUT_S = 900  # 15 min.
# Tightened from 1800s after multi-node sessions observed variants
# silent-aborting after exactly 30 min of sglang launch-poll RUNNING —
# the variant's config was incompatible with the model's multi-node
# cold-start path and the launcher driver never reached /health 200.
# Typical multi-node MoE cold-start finishes in 5-7 min, so 900s is
# ~2x normal headroom and still aborts ~2x faster than the old 1800s
# ceiling. Override per-run via HYPERLOOM_MN_HEALTH_WAIT_S when a
# slower workload genuinely needs more — the env override path in
# restart_server_for_round is preserved.

# Defaults Magpie's sglang_mi*x.sh always appends to the server cmd when
# the variant did not explicitly override them. Keeping the SAME flag set
# in multi-node mode is what makes tput/accuracy numbers comparable to
# the single-node baseline; the *value* for `--mem-fraction-static` is
# the one place we deliberately diverge:
#
#   * single-node `sglang_mi*x.sh:74` ships --mem-fraction-static=0.8
#   * multi-node we lower to 0.75 because cross-node RDMA / RoCE
#     buffers (mlx5 pinned memory, NCCL connection pool) eat noticeably
#     more headroom than the single-node intra-pod NVLink/XGMI path.
#     0.8 has been observed to OOM on DSr1 671B FP8 when the second
#     node joins.
#
# Shape: (flag_name, full_token). ``_merge_sglang_defaults`` skips a
# default if the user's extra_args already mentions ``flag_name``
# (matching Magpie's own behaviour at ``sglang_mi300x.sh:74-79``).
_SGLANG_DEFAULT_TOKENS: tuple[tuple[str, str], ...] = (
    ("--mem-fraction-static", "--mem-fraction-static=0.75"),
    ("--disable-radix-cache", "--disable-radix-cache"),
)


def _merge_sglang_defaults(extra_args: str) -> str:
    """Append Magpie's DEFAULT_ARGS that the user did not already set.

    Mirrors ``Magpie/scripts/benchmark/sglang_mi300x.sh:74-79`` so the
    multi-node sglang server gets the same conservative tuning baseline
    as single-node when the variant did not provide an explicit value.

    For example, with ``extra_args=""`` returns
    ``"--mem-fraction-static=0.8 --disable-radix-cache"``; with
    ``extra_args="--mem-fraction-static=0.9"`` returns
    ``"--mem-fraction-static=0.9 --disable-radix-cache"`` (caller's
    explicit override wins, the default is dropped).

    Args:
        extra_args (str): The variant's explicit server args (may be empty).

    Returns:
        str: ``extra_args`` with any unset Magpie default tokens appended.
    """
    user = (extra_args or "").strip()
    parts = [user] if user else []
    for flag_name, default_token in _SGLANG_DEFAULT_TOKENS:
        if flag_name in user:
            continue
        parts.append(default_token)
    return " ".join(p for p in parts if p)

# Persist round-trip context here so callers (profile.py) can recover
# the path the server was actually restarted with, even after this
# helper restored the previous env. Read-only for callers.
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

    Returns a flat dict of resolved PD values; callers pass it to the
    multi_node CLI argparse Namespace. Validates ``pd_mode`` and (when
    disaggregated) the prefill/decode TP <= per-group capacity.

    Resolution order per field:
      1. Explicit kwarg (executor-supplied, usually from $PD_* env).
      2. ``state["last_restart_pd_*"]`` from prior restart.
      3. Process env (``$PD_MODE`` etc.).
      4. Defaults (mode=colocated, prefill/decode TP=tp).

    Args:
        pd_mode (str | None): ``"colocated"`` / ``"disaggregated"`` override.
        pd_prefill_nodes (int | None): Prefill-group node count override.
        pd_decode_nodes (int | None): Decode-group node count override.
        pd_prefill_tp (int | None): Prefill-group tensor-parallel override.
        pd_decode_tp (int | None): Decode-group tensor-parallel override.
        pd_transfer_backend (str | None): KV transfer backend override.
        pd_ib_device (str | None): InfiniBand device override.
        tp_int (int): The resolved overall tensor-parallel degree, used as the
            prefill/decode TP default.

    Returns:
        dict: Resolved PD values (``pd_mode`` plus, for disaggregated mode, the
            node / TP / backend / IB fields) for the multi_node CLI namespace.

    Raises:
        ServerRestartFailed: If ``pd_mode`` is unsupported or the resolved
            disaggregated node / TP values are invalid.
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

    # PD disaggregation requires >=2 nodes. is_multi_node() already
    # gated the helper entry, but defend in depth: if state.json got
    # mangled (e.g. provisioning re-ran with --nodes 1 mid-session)
    # we'd otherwise try to split a single pod into two roles. Catch
    # the inconsistency here and surface it as a recoverable round
    # failure rather than a confusing sglang launcher error.
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

    Resolution order per field:
      1. Explicit kwarg (executor-supplied, usually from task.params).
      2. ``state["last_restart_*"]`` from prior successful restart.
      3. Process env (``$FRAMEWORK`` / ``$MODEL_PATH`` / ``$TP`` / ``$EP``).
      4. ep specifically: defaults to ``1`` (no EP) when nothing supplied,
         keeping the legacy TP-shard-experts behaviour.

    Raises :class:`ServerRestartFailed` if model/tp end up empty, if
    framework is unsupported, or if ``ep > tp`` (sglang / vllm cannot
    place more expert shards than ranks).

    Args:
        framework (str | None): Framework override (``sglang`` / ``vllm``).
        model_path (str | None): Model path override.
        tp (int | None): Tensor-parallel degree override.
        ep (int | None): Expert-parallel degree override (defaults to 1).

    Returns:
        tuple[str, str, int, int]: ``(framework, model, tp, ep)`` resolved
            values.

    Raises:
        ServerRestartFailed: If model/tp are missing, the framework is
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

    No-op (returns immediately) when ``is_multi_node()`` is False so the
    single-node code path is preserved bit-for-bit.

    For multi-node:
      * Resolves framework/model/tp (kwargs > state.json > env).
      * Mkdir ``torch_profiler_dir`` (when non-empty) and exports it via
        ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` so the existing
        ``multi_node/cli.py _build_multinode_launch_entrypoint`` picks
        it up and forwards ``--torch-profiler-dir <dir>`` to every
        sglang launcher rank.
      * Invokes ``cmd_restart_server`` synchronously inside a thread
        (the function does Ray Dashboard polling internally).
      * Restores the previous env value of
        ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` afterwards so a later round
        without a profile dir doesn't accidentally inherit this round's
        path.

    ``force_full_restart``: when True, scopes
    ``MULTI_NODE_RESTART_RESUME_RUNNING=0`` for this single invocation
    so cmd_restart_server's resume fast-path is bypassed and a fresh
    kill+launch always runs. Required by the kernel-agent integrate
    path: after apply_kernel_patch fans new source files to every pod,
    sglang must be fully restarted (not just /flush_cache resumed) to
    re-import the patched modules. The previous env value is restored
    in ``finally`` so non-kernel-opt callers keep their resume savings.

    Raises :class:`ServerRestartFailed` on any failure; callers should
    let it bubble so the round is marked failed.

    Args:
        extra_server_args (str): Server-side flags for this round's launch.
        torch_profiler_dir (str): Per-round profiler trace dir; exported so the
            launcher forwards ``--torch-profiler-dir``. Empty disables it.
        framework (str | None): Framework override (``sglang`` / ``vllm``).
        model_path (str | None): Model path override.
        tp (int | None): Tensor-parallel degree override.
        ep (int | None): Expert-parallel degree override.
        pd_mode (str | None): PD mode override (``colocated`` /
            ``disaggregated``).
        pd_prefill_nodes (int | None): Prefill-group node count override.
        pd_decode_nodes (int | None): Decode-group node count override.
        pd_prefill_tp (int | None): Prefill-group TP override.
        pd_decode_tp (int | None): Decode-group TP override.
        pd_transfer_backend (str | None): KV transfer backend override.
        pd_ib_device (str | None): InfiniBand device override.
        health_timeout_s (int): Timeout budget for the post-launch /health
            wait and launch-driver poll.
        poll_interval_s (int): Poll interval (seconds) for the launch driver.
        force_full_restart (bool): When True, bypass the resume fast-path so a
            fresh kill+launch always runs (needed after kernel patches).

    Raises:
        ServerRestartFailed: On any restart / health-wait failure.
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

    # PD-disaggregated × EP cross-check: ep > min(prefill_tp, decode_tp)
    # would put more expert shards than ranks on at least one side.
    if pd["pd_mode"] == "disaggregated" and ep_int > 1:
        min_grp_tp = min(pd["pd_prefill_tp"], pd["pd_decode_tp"])
        if ep_int > min_grp_tp:
            raise ServerRestartFailed(
                f"ep={ep_int} > min(pd_prefill_tp={pd['pd_prefill_tp']}, "
                f"pd_decode_tp={pd['pd_decode_tp']})={min_grp_tp}; "
                "lower --ep or raise the smaller per-group TP."
            )

    # Apply Magpie's sglang DEFAULT_ARGS only for sglang; vllm has its
    # own defaults baked into the server and Magpie doesn't append the
    # same flags there.
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
        # Round has no profiler — drop any stale env so the launcher
        # doesn't reuse a previous round's path on this restart.
        os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)
        _LAST_ROUND_TRACE_DIR = ""

    # Multi-node TraceLens SGLang patch fan-out (fail-soft).
    #
    # Single-node ``ensure_sglang_patched_for_tracelens`` runs in the
    # same Python process that imports SGLang; multi-node SGLang lives
    # in head/worker pods so the controller cannot ``import sglang`` and
    # the local patcher silently skips. Without these patches SGLang's
    # torch.profiler emits step boundaries the TraceLens splitter does
    # NOT recognise (``step[DECODE bs=N]`` instead of vLLM-style
    # ``execute_*_context_*_generation_*``), so every profile round
    # ends in ``trace_split_no_steady_state`` and the orchestration
    # agent stalls. We invoke the multi-node fan-out (idempotent: each
    # pod sentinel-greps before applying) BEFORE ``cmd_restart_server``
    # so the restarted SGLang already has the patches in place.
    #
    # Fail-soft: if the patch fan-out fails (TraceLens missing, ssh
    # error, version unsupported, ...) we log a warning and proceed with
    # the restart anyway. The trace will be unannotated and tracelens
    # analysis will surface the splitter warning, but every other phase
    # (baseline / grid / validate_stack / kernel) keeps working — far
    # better than blocking the entire restart on what is supposed to be
    # an opt-in profiling enhancement.
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
        # Local import: avoid pulling httpx into the import path of any
        # caller that doesn't actually invoke the helper (single-node).
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
            # PD knobs forwarded to multi_node CLI; colocated mode passes
            # only pd_mode and the rest stay at argparse defaults.
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

        # When kernel-agent just patched sglang source on every pod, the
        # resume fast-path (MULTI_NODE_RESTART_RESUME_RUNNING) would
        # leave the still-running sglang process holding the OLD module
        # imports and the patch would have no effect. Scope an env
        # override for THIS invocation only so cmd_restart_server's
        # prev_match check fails and a fresh kill+launch runs.
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

        # ADDENDUM: cmd_restart_server returns when the launcher driver
        # finishes spawning actors (servers detached); on a cold MoE
        # weight-load (DeepSeek-R1 671B) the server itself can need
        # 20-30 min before /health flips. The downstream baseline
        # benchmark fires immediately and 100%-fails if we don't wait
        # here. Poll the service_url /health (head-pod IP) until it
        # 200's or we hit health_timeout_s.
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
        # Restore previous env so we don't leak this round's profiler
        # path into the orchestrator process or subsequent helper calls.
        if saved_trace_env is None:
            os.environ.pop("HYPERLOOM_MN_PROFILE_TRACE_DIR", None)
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = saved_trace_env




async def _wait_for_server_health_async(
    timeout_s: int = 1800,
    poll_every_s: int = 10,
) -> None:
    """Poll the multi_node service_url /health until 200 or timeout.

    Reads ``service_url`` from ``state.json`` (the multi_node CLI's
    persisted RayJob state). When the URL points at the in-cluster
    ClusterIP DNS name and we also have a ``head_pod_ip`` we rewrite
    to the head-pod IP so the sandbox can reach it directly.

    Args:
        timeout_s (int): Maximum time to wait for a 200 from /health.
        poll_every_s (int): Interval between /health polls, in seconds.

    Raises:
        ServerRestartFailed: If /health does not return 200 within
            ``timeout_s``.
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
