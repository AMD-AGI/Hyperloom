# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``inference_optimizer.multi_node`` — single-entry CLI used by the agent
inside the Claw sandbox to manage one session-scoped RayJob for the whole
optimization session.

Subcommands (run via ``python3 -m inference_optimizer.multi_node <sub>``):

    create-rayjob    Create the RayJob via SaFE REST and wait for it to be
                     Running. Checkpoints ``rayjob_id`` to
                     ``/tmp/multi_node_state.json`` immediately after SaFE
                     returns an id so parallel retries never create a second
                     workload. Then polls until ``phase=Running`` and fills
                     ``head_pod_ip`` / ``service_url``.

    bootstrap        Submit ``$BOOTSTRAP_SCRIPT`` to the head pod via the
                     Ray Dashboard REST API. Polls until the submission
                     reaches a terminal status.

    verify           Submit a tiny ``which oob && which claude && ...``
                     script via Ray Dashboard REST. Used right after
                     bootstrap to confirm the toolchain is ready.

    restart-server   Submit a shell script that kills the previous
                     vllm/sglang process (via PID file) and launches a
                     new one in the background, then waits for /health.
                     The Ray Dashboard job exits as soon as the launch
                     command returns; the actual server keeps running
                     because it was nohup'd. Idempotent: repeated calls
                     replace the previous server.

    kill-inference   Kill vllm/sglang only (no new server). Frees GPUs on
                     the RayJob before kernel-agent submits Ray GPU tasks.

    stop-rayjob      Soft-stop the RayJob via SaFE REST. Idempotent.

State file (``/tmp/multi_node_state.json``) shape::

    {
      "rayjob_id":          "<workload-id>",
      "workspace":          "<safe-workspace>",
      "head_pod_ip":        "10.x.x.x",
      "ray_dashboard_url":  "http://10.x.x.x:8265",
      "service_url":        "http://<wid>.<ws>.svc.cluster.local:8888",
      "last_server_pid_file": "/wekafs/.../server.pid",
      "ray_address":          "<head_pod_ip>:6379",
      "last_create_request": { ...echoed back to allow inspection... }
    }

Operational constraints baked in (see also multi_node/SKILL.md):

* Every wait loop here uses many short HTTP polls instead of one long
  ``sleep`` — see ADDENDUM-09. The sandbox bash that invokes us has a
  120s ceiling per call; we surface progress on stderr every poll so
  the agent sees forward motion.

* Credentials (SAFE_API_URL/SAFE_API_KEY/OOB_BASE_URL/...) must already
  exist in sandbox env before the CLI is invoked — see ADDENDUM-13.
  This module never invents URLs or keys.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..paths import mn_profile_trace_root
from ._internal import safe_client, ray_dashboard, workload_spec
from ._internal import ssh_client, dynamo_support
from ._internal.log import info, warn, err

STATE_FILE = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))

# Defaults sized so a single bash invocation of the CLI never exceeds
# the sandbox 120s ceiling (ADDENDUM-09): poll every 6s for up to 18
# attempts ≈ 110s. Caller can shrink by passing a stricter budget.
_DEFAULT_POLL_INTERVAL_S = 6
_DEFAULT_POLL_TIMEOUT_S = 110
# MoE cold-start (weight load + aiter JIT) often needs 20-30 min on
# multi-node RayJobs. Export HYPERLOOM_MN_POLL_TIMEOUT_S=1800 (and
# HYPERLOOM_MN_HEALTH_WAIT_S=1800 for /health) before optimize / restart.
_DEFAULT_JIT_POLL_TIMEOUT_S = 1800


def _resolve_poll_timeout_s() -> int:
    """Poll budget for one multi_node CLI invocation (seconds).

    Resolution: ``HYPERLOOM_MN_POLL_TIMEOUT_S`` env > ``_DEFAULT_POLL_TIMEOUT_S``.
    Large-model / JIT runs should set 1800 (30 min) in the launcher env.
    """
    raw = (os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            warn(
                f"invalid HYPERLOOM_MN_POLL_TIMEOUT_S={raw!r}; "
                f"using {_DEFAULT_POLL_TIMEOUT_S}"
            )
    return _DEFAULT_POLL_TIMEOUT_S


def _poll_timeout_from_args(args: argparse.Namespace) -> int:
    """CLI flag wins; else env/default from :func:`_resolve_poll_timeout_s`."""
    pt = getattr(args, "poll_timeout", None)
    if pt is not None:
        return max(1, int(pt))
    return _resolve_poll_timeout_s()


# SaFE may return 404 on GET /workloads/{id} briefly after POST create returns
# an id (read-after-write lag). Treat as benign for this window only.
_SAFE_GET_WORKLOAD_404_GRACE_S = 30.0

# RayJob phase strings reported by SaFE.
_TERMINAL_FAIL_PHASES = {"Failed", "Stopped", "Cancelled"}
_TERMINAL_OK_PHASES = {"Running"}

# Ray dashboard job status strings.
_TERMINAL_FAIL_STATUSES = {"FAILED", "STOPPED"}
_TERMINAL_OK_STATUSES = {"SUCCEEDED"}


def _normalize_extra_args(s: str | None) -> str:
    """Normalize sglang/vllm ``--extra-args`` for equality comparison.

    Collapses arbitrary whitespace (leading/trailing/multi-space) to single
    spaces so semantically-identical arg strings produced by different
    callers compare equal. Order-sensitive — argv order may matter to the
    framework (e.g. last-wins for repeated flags) so we deliberately do
    NOT sort.
    """
    return " ".join((s or "").split())

# Exit codes the hyperloom main controller can switch on. Keep these
# stable; they are part of the CLI's contract with the agent.
EXIT_OK = 0
EXIT_TRANSIENT = 1          # network / SaFE 5xx / timeout — caller may retry
EXIT_TERMINAL_FAILURE = 2   # workload entered Failed/Stopped/Cancelled — DO NOT retry; fix and recreate
EXIT_CONFIG_ERROR = 3       # missing env / required arg — fix the call, don't retry blindly
EXIT_INTERRUPT = 130        # Ctrl-C / SIGINT


class WorkloadTerminalFailure(RuntimeError):
    """Raised when SaFE reports the workload has entered a terminal failure
    phase (Failed / Stopped / Cancelled). Carries the diagnostic snapshot
    for the hyperloom controller to log / act on. Exit code -> 2.
    """

    def __init__(self, label: str, phase: str, diag: str, snapshot: dict[str, Any]) -> None:
        super().__init__(f"{label} terminal phase={phase}: {diag}")
        self.label = label
        self.phase = phase
        self.diag = diag
        self.snapshot = snapshot


class TransientFailure(RuntimeError):
    """Raised on poll timeout or repeated SaFE communication failure.
    Exit code -> 1; caller may rerun the same subcommand.
    """


# ---------------------------------------------------------------------------
# State file


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"state file {STATE_FILE} unreadable: {exc}")
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_create_rayjob_state(
    *,
    wid: str,
    workspace: str,
    args: argparse.Namespace,
) -> None:
    """Persist ``rayjob_id`` as soon as SaFE returns an id (or we reuse one).

    Without this, a second ``create-rayjob`` started while the first is still
    polling for ``phase=Running`` would not see ``/tmp/multi_node_state.json``
    yet and would call ``create_workload`` again, leaking a second RayJob per
    sandbox. See SKILL idempotency / one-RayJob-per-session rule.
    """
    prev = _load_state()
    old = (prev.get("rayjob_id") or "").strip()
    state: dict[str, Any] = dict(prev)
    if old and old != wid:
        for k in ("head_pod_ip", "ray_dashboard_url", "last_bootstrap_submission_id"):
            state.pop(k, None)
    state["rayjob_id"] = wid
    state["workspace"] = workspace
    state["nodes"] = args.nodes
    state["gpus_per_node"] = args.gpus_per_node
    state["service_url"] = f"http://{wid}.{workspace}.svc.cluster.local:8888"
    state["last_create_request"] = {
        "image": args.image,
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
    }
    if old != wid:
        state["head_pod_ip"] = ""
        state["ray_dashboard_url"] = ""
    else:
        state.setdefault("head_pod_ip", "")
        state.setdefault("ray_dashboard_url", "")
    state.setdefault("ray_address", "")
    _save_state(state)
    info(f"checkpointed rayjob_id={wid} to {STATE_FILE}")


def _write_rayjob_meta(
    *,
    wid: str,
    workspace: str,
    session_id: str | None,
    owner_id: str | None,
    display_name: str,
    nodes: int,
    gpus_per_node: int,
) -> None:
    """Drop a per-session meta JSON under the multi-node profile-trace root.

    Path: ``<workspace_root>/profile-traces/<rayjob_id>/<session_id>``.

    The file ties this RayJob workload back to the sandbox session that
    created it so operators can correlate ``profile-traces/<wid>/torch_trace/``
    artefacts (already produced under the same ``<wid>/`` parent by
    ``launch_multinode.py``) with the originating Claw session without
    walking the SaFE API. Co-locating meta with traces means a single
    ``rsync`` of ``profile-traces/<wid>/`` carries both the data and its
    provenance.

    Skipped entirely when ``session_id`` is empty / None: the file name
    is the session id itself, so a missing id leaves us nowhere to write.
    This mirrors the label-injection skip rule in :func:`cmd_create_rayjob`.

    Best-effort: any filesystem error (read-only mount, quota, permission)
    is logged at WARN and swallowed. Meta is audit data, not a critical
    path of RayJob creation; failing the workload because we couldn't
    write a sidecar JSON would be a worse outcome than missing the meta.
    """
    if not session_id:
        return
    meta_path = mn_profile_trace_root() / wid / session_id
    payload: dict[str, Any] = {
        "rayjob_id": wid,
        "session_id": session_id,
        "owner_id": owner_id,
        "workspace": workspace,
        "display_name": display_name,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        info(f"rayjob meta written to {meta_path}")
    except OSError as exc:
        warn(f"failed to write rayjob meta to {meta_path}: {exc}")


def _require_state(*keys: str) -> dict[str, Any]:
    """Load state and assert all required keys are present."""
    state = _load_state()
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise RuntimeError(
            f"State file {STATE_FILE} missing required keys: {missing}. "
            f"Have you run 'create-rayjob' first?"
        )
    return state


# ---------------------------------------------------------------------------
# Helpers


def ray_gcs_address(head_pod_ip: str) -> str:
    """Ray driver address for ``ray.init(address=...)`` (GCS on head, default port)."""
    ip = (head_pod_ip or "").strip()
    if not ip:
        return ""
    return f"{ip}:6379"


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _wrap_for_dash(body: str) -> str:
    """Wrap a bash entrypoint so it survives Ray Dashboard /api/jobs/ exec under
    /bin/sh (dash) on the sglang:202604290707 image. Dash rejects
    ``set -o pipefail``; bash accepts it. We base64-encode the body and pipe it
    into bash so the outer shell only needs ``echo``/``base64``/``bash``.
    """
    enc = _b64(body)
    return f"echo {enc} | base64 -d | bash"


def _parse_kv_list(values: list[str] | None) -> dict[str, str]:
    """Convert ['K=V', 'K2=V2', ...] into a dict; ignore malformed entries."""
    out: dict[str, str] = {}
    if not values:
        return out
    for raw in values:
        if "=" not in raw:
            warn(f"ignoring malformed K=V token: {raw!r}")
            continue
        k, _, v = raw.partition("=")
        k = k.strip()
        if not k:
            continue
        out[k] = v
    return out


def _credential_fanout() -> dict[str, str]:
    """Return the *_API_KEY / *_BASE_URL env to inject into the RayJob.

    Source per ADDENDUM-13: every key falls back to ``SAFE_API_KEY`` if
    not explicitly set in sandbox env. ``ANTHROPIC_BASE_URL`` /
    ``OOB_BASE_URL`` / ``ANTHROPIC_CUSTOM_HEADERS`` are passed through
    as-is from sandbox env (they are cluster-specific values that the
    SaFE platform sets on the sandbox at start time).
    """
    safe_key = os.environ.get("SAFE_API_KEY", "").strip()
    out: dict[str, str] = {}
    for name in (
        "OOB_API_KEY",
        "AMD_LLM_API_KEY",
        "LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "SAFE_API_KEY",
    ):
        v = os.environ.get(name, "").strip()
        if not v and safe_key:
            v = safe_key
        if v:
            out[name] = v
    for name in ("OOB_BASE_URL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
        v = os.environ.get(name, "").strip()
        if v:
            out[name] = v
    return out


def _is_safe_get_workload_404(exc: BaseException) -> bool:
    """True when SaFE GET workload returns 404 (transient right after create)."""
    return (
        isinstance(exc, safe_client.SafeApiError)
        and exc.status == 404
        and "GET /api/v1/workloads/" in (exc.endpoint or "")
    )


def _short_poll(
    *,
    label: str,
    fetch: callable,
    is_ok: callable,
    is_fail: callable,
    interval_s: int,
    timeout_s: int,
    failure_diag: callable | None = None,
    quiet_fetch_error_grace_s: float = 0.0,
    is_quiet_fetch_error: Callable[[BaseException], bool] | None = None,
) -> Any:
    """Run many short polls within a single CLI invocation budget.

    ``fetch()`` returns a (state_obj, summary_str) tuple. ``is_ok`` /
    ``is_fail`` inspect ``state_obj``. We log every poll to stderr so the
    agent sees forward motion. Returns the final state_obj on success.

    When ``quiet_fetch_error_grace_s > 0`` and ``is_quiet_fetch_error(exc)``
    is true for a fetch exception, and elapsed time since poll start is
    within the grace window, log at INFO instead of WARN and omit the
    duplicate summary line (used for post-create GET 404 lag on SaFE).

    On terminal failure: if ``failure_diag(obj)`` is provided, raise
    :class:`WorkloadTerminalFailure` with a structured snapshot the
    hyperloom controller can switch on (exit code 2). Otherwise raise
    a plain :class:`RuntimeError` with just the summary line.

    On poll timeout: raise :class:`TransientFailure` (exit code 1) so
    the controller knows it's safe to rerun the same subcommand to keep
    waiting.
    """
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        fetch_was_quiet = False
        try:
            obj, summary = fetch()
        except Exception as exc:  # noqa: BLE001
            elapsed_mid = time.monotonic() - started
            fetch_was_quiet = (
                quiet_fetch_error_grace_s > 0.0
                and elapsed_mid < quiet_fetch_error_grace_s
                and is_quiet_fetch_error is not None
                and is_quiet_fetch_error(exc)
            )
            if fetch_was_quiet:
                info(
                    f"{label} poll #{attempt} t={elapsed_mid:.0f}s -> "
                    f"GET workload not visible yet (404); retrying "
                    f"(expected within ~{quiet_fetch_error_grace_s:.0f}s after create)"
                )
            else:
                warn(f"{label} poll #{attempt} fetch error: {exc}")
            obj, summary = None, str(exc)

        elapsed = time.monotonic() - started
        if not fetch_was_quiet:
            info(f"{label} poll #{attempt} t={elapsed:.0f}s -> {summary}")

        if obj is not None:
            if is_ok(obj):
                return obj
            if is_fail(obj):
                if failure_diag is not None:
                    diag, snapshot = failure_diag(obj)
                    raise WorkloadTerminalFailure(
                        label=label,
                        phase=str(obj.get("phase") if isinstance(obj, dict) else "?"),
                        diag=diag,
                        snapshot=snapshot,
                    )
                raise RuntimeError(f"{label} terminal failure: {summary}")

        if elapsed >= timeout_s:
            raise TransientFailure(
                f"{label} did not reach a terminal state within {timeout_s}s "
                f"(last: {summary}). Re-run the same subcommand to keep polling."
            )
        time.sleep(interval_s)


def _summarize_workload_failure(workload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a human-readable single-line diagnostic and a structured
    snapshot from a SaFE GetWorkloadResponse, suitable for terminal-
    failure errors. Mirrors what the brain TS code logs (phase + message
    + per-pod status + dispatch count) so the hyperloom controller can
    decide whether to give up vs surface a useful error.

    Returns (one_line_summary, snapshot_dict). Snapshot is JSON-safe.
    """
    phase = str(workload.get("phase") or "?")
    msg = str(workload.get("message") or "").strip()
    queue = workload.get("queuePosition")
    dispatch = workload.get("dispatchCount")
    pods = workload.get("pods") or []
    pod_summary: list[dict[str, Any]] = []
    for p in pods:
        if not isinstance(p, dict):
            continue
        pod_summary.append({
            "podId": p.get("podId"),
            "phase": p.get("phase"),
            "resourceId": p.get("resourceId"),
            "node": p.get("adminNodeName"),
            "podIP": p.get("podIP"),
            "failedMessage": p.get("failedMessage"),
        })

    parts = [f"phase={phase}"]
    if msg:
        parts.append(f"message={msg!r}")
    if dispatch is not None:
        parts.append(f"dispatchCount={dispatch}")
    if queue is not None and phase in ("Pending", "Updating"):
        parts.append(f"queuePosition={queue}")
    bad_pods = [p for p in pod_summary if p.get("phase") in ("Failed", "Unknown") or p.get("failedMessage")]
    if bad_pods:
        # Show up to first 3 failing pods inline; full list lives in snapshot.
        bp_strs = []
        for bp in bad_pods[:3]:
            bp_strs.append(
                f"{bp.get('podId') or '?'}({bp.get('phase') or '?'}"
                + (f": {bp['failedMessage']}" if bp.get('failedMessage') else "")
                + ")"
            )
        parts.append("failed_pods=[" + ", ".join(bp_strs)
                     + (", ..." if len(bad_pods) > 3 else "") + "]")
    elif pod_summary:
        parts.append(f"pods={len(pod_summary)}")
    diag = " ".join(parts)
    snapshot = {
        "workloadId": workload.get("workloadId"),
        "phase": phase,
        "message": msg,
        "dispatchCount": dispatch,
        "queuePosition": queue,
        "pods": pod_summary,
    }
    return diag, snapshot


def _find_head_pod_ip(workload: dict) -> str:
    """Pick the Ray **KubeRay head** pod PodIp from a GetWorkloadResponse.

    SaFE may list the RayJob submitter pod with ``resourceId == 0``; that
    pod is not the Ray head and often has a host-network IP that does not
    expose port 8265. KubeRay head pod names always contain the substring
    ``-head-`` (e.g. ``...-x6fkf-head-ddtvg``).

    Priority:
      1. ``podId`` contains ``-head-`` and ``podIP`` is set.
      2. ``resourceId == 0`` (legacy when submitter was not resource 0).
      3. first pod with non-empty ``podIP``.
    """
    pods = workload.get("pods") or []
    if not pods:
        return ""
    for p in pods:
        pid = str(p.get("podId") or "")
        if "-head-" in pid and p.get("podIP"):
            return str(p["podIP"])
    for p in pods:
        if p.get("resourceId") == 0 and p.get("podIP"):
            return str(p["podIP"])
    for p in pods:
        if p.get("podIP"):
            return str(p["podIP"])
    return ""


# ---------------------------------------------------------------------------
# Subcommand: create-rayjob


def cmd_create_rayjob(args: argparse.Namespace) -> int:
    """Create the RayJob, checkpoint ``rayjob_id`` early, then poll for Running."""
    extra_env = _parse_kv_list(args.extra_env)
    extra_labels = _parse_kv_list(args.extra_label)
    # User extra_env takes precedence over our credential fanout, except
    # for keys workload_spec marks as reserved.
    env = {**_credential_fanout(), **extra_env}

    # ownerId fallback chain:
    #   --owner-id > $WORKLOAD_ID
    #
    # $WORKLOAD_ID is the sandbox SaFE workload id (Brain-injected). SaFE
    # cascades: when that owner workload stops, workloads listing it as
    # ownerId are GC'd. Omit ownerId when neither flag nor env is set.
    owner_id = (
        args.owner_id
        or os.environ.get("WORKLOAD_ID", "").strip()
        or None
    )
    if owner_id and not args.owner_id:
        info(f"ownerId derived from $WORKLOAD_ID: {owner_id}")

    # workspace resolution: --workspace > $SAFE_WORKSPACE env. The CLI
    # bails fast (clear RuntimeError) if neither is set so the agent
    # gets a single human-readable line, not a SaFE 400 error.
    workspace = (args.workspace or os.environ.get("SAFE_WORKSPACE", "").strip())
    if not workspace:
        raise RuntimeError(
            "workspace is required: pass --workspace <safe-workspace> "
            "or export $SAFE_WORKSPACE in the sandbox env. "
            "Brain normally sets this at sandbox startup."
        )
    if not args.workspace:
        info(f"workspace derived from $SAFE_WORKSPACE: {workspace}")

    # display_name resolution: $DISPLAY_NAME env > --display-name > fallback.
    display_name = (
        os.environ.get("DISPLAY_NAME", "").strip()
        or args.display_name
        or f"multi_node_{int(time.time())}"
    )
    info(f"displayName: {display_name}")

    # session_id: read from $CLAW_SESSION_ID (Brain-injected at sandbox
    # start). When unset we skip the label so dev / local runs without
    # Brain don't fail; production sandboxes always have it exported.
    session_id = (os.environ.get("CLAW_SESSION_ID") or "").strip() or None
    if session_id:
        info(f"sessionId derived from $CLAW_SESSION_ID: {session_id}")

    body = workload_spec.build_rayjob_workload_body(
        workspace=workspace,
        display_name=display_name,
        image=args.image,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        cpus_per_node=args.cpus_per_node,
        mem_gi_per_node=args.mem_per_node,
        ephemeral_gi_per_node=args.ephemeral_per_node,
        description=args.description,
        owner_id=owner_id,
        session_id=session_id,
        extra_env=env,
        extra_labels=extra_labels,
    )

    with safe_client.from_env() as safe:
        # Idempotency guard: if state already has a ``rayjob_id`` (written
        # immediately after the last successful ``create_workload`` or reuse)
        # and that workload is still non-terminal in SaFE, REUSE it. Otherwise
        # we'd leak one workload per overlapping ``create-rayjob`` invocation.
        # Pass --recreate to force a fresh workload regardless of state.
        wid: str | None = None
        existing = _load_state()
        prior_wid = (existing.get("rayjob_id") or "").strip()
        if prior_wid and not getattr(args, "recreate", False):
            try:
                prior_wl = safe.get_workload(prior_wid)
            except safe_client.SafeApiError as exc:
                if exc.status == 404:
                    info(
                        f"prior rayjob_id={prior_wid} no longer exists in SaFE; "
                        f"will create a fresh workload"
                    )
                else:
                    raise
            else:
                prior_phase = str(prior_wl.get("phase") or "?")
                if prior_phase in _TERMINAL_FAIL_PHASES:
                    info(
                        f"prior rayjob_id={prior_wid} is in terminal phase "
                        f"{prior_phase!r}; will create a fresh workload"
                    )
                else:
                    info(
                        f"reusing existing rayjob_id={prior_wid} from state file "
                        f"(phase={prior_phase}); will resume polling instead of "
                        f"creating a new workload"
                    )
                    wid = prior_wid

        if wid is None:
            info(f"creating RayJob workload (workspace={workspace} nodes={args.nodes})")
            wid = safe.create_workload(body)
            info(f"workload created: {wid}")
            _write_rayjob_meta(
                wid=wid,
                workspace=workspace,
                session_id=session_id,
                owner_id=owner_id,
                display_name=display_name,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
            )

        _checkpoint_create_rayjob_state(wid=wid, workspace=workspace, args=args)

        head_pod_ip = ""
        if args.no_wait:
            info("--no-wait set; not polling for Running")
        else:
            def _fetch():
                wl = safe.get_workload(wid)
                phase = wl.get("phase", "?")
                summary = f"phase={phase}"
                return wl, summary

            workload = _short_poll(
                label=f"workload {wid}",
                fetch=_fetch,
                is_ok=lambda w: w.get("phase") in _TERMINAL_OK_PHASES,
                is_fail=lambda w: w.get("phase") in _TERMINAL_FAIL_PHASES,
                interval_s=args.poll_interval,
                timeout_s=_poll_timeout_from_args(args),
                failure_diag=_summarize_workload_failure,
                quiet_fetch_error_grace_s=_SAFE_GET_WORKLOAD_404_GRACE_S,
                is_quiet_fetch_error=_is_safe_get_workload_404,
            )
            head_pod_ip = _find_head_pod_ip(workload)
            if not head_pod_ip:
                warn(
                    "workload Running but no head pod IP yet; SaFE may need "
                    "another sync. Re-run create-rayjob to refresh."
                )

    merged = dict(_load_state())
    merged.update(
        {
            "rayjob_id": wid,
            "workspace": workspace,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "head_pod_ip": head_pod_ip,
            "ray_address": ray_gcs_address(head_pod_ip),
            "ray_dashboard_url": (
                ray_dashboard.dashboard_url(head_pod_ip) if head_pod_ip else ""
            ),
            "service_url": f"http://{wid}.{workspace}.svc.cluster.local:8888",
            "last_create_request": {
                "image": args.image,
                "nodes": args.nodes,
                "gpus_per_node": args.gpus_per_node,
            },
        }
    )
    _save_state(merged)
    info(f"state written to {STATE_FILE}")
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: create-dynamo (Dynamo idle-pod backend)
#
# Mirrors create-rayjob but provisions a SaFE DynamoDeployment with idle
# worker pods (mn-idle.sh) and an SSH control plane instead of a RayJob with
# the Ray Dashboard. The benchmark entry point is the Dynamo frontend
# (:8000), NOT sglang rank-0 :8888.

# Session-scoped SSH key dir (sandbox-local, next to the state file).
_DYNAMO_SSH_DIR = STATE_FILE.parent / "mn_ssh"


def cmd_create_dynamo(args: argparse.Namespace) -> int:
    """Create an idle multi-node DynamoDeployment, then poll for Running.

    Generates a session SSH keypair (public key -> MN_SSH_AUTHORIZED_KEY in the
    workload body), creates/reuses the workload, waits for Running, then
    discovers the worker pod IPs and the frontend service URL into the state
    file so restart-server can SSH-fan-out the inference server.
    """
    extra_env = _parse_kv_list(args.extra_env)
    extra_labels = _parse_kv_list(args.extra_label)
    env = {**_credential_fanout(), **extra_env}

    owner_id = (
        args.owner_id or os.environ.get("WORKLOAD_ID", "").strip() or None
    )
    workspace = (args.workspace or os.environ.get("SAFE_WORKSPACE", "").strip())
    if not workspace:
        raise RuntimeError(
            "workspace is required: pass --workspace or export $SAFE_WORKSPACE"
        )
    display_name = (
        os.environ.get("DISPLAY_NAME", "").strip()
        or args.display_name
        or f"dynamo_mn_{int(time.time())}"
    )
    session_id = (os.environ.get("CLAW_SESSION_ID") or "").strip() or None

    # Session SSH keypair: public key authorises the controller on every pod.
    priv_key, pub_key = ssh_client.generate_session_keypair(_DYNAMO_SSH_DIR)

    pd_mode = (getattr(args, "pd_mode", "") or "aggregated").lower()
    body = workload_spec.build_dynamo_workload_body(
        workspace=workspace,
        display_name=display_name,
        image=args.image,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        cpus_per_node=args.cpus_per_node,
        mem_gi_per_node=args.mem_per_node,
        ephemeral_gi_per_node=args.ephemeral_per_node,
        ssh_authorized_key=pub_key,
        backend_framework=args.backend_framework,
        kv_transfer_backend=args.kv_transfer_backend,
        ssh_port=args.ssh_port,
        shared_mem_gi=args.shared_mem_per_node,
        pd_mode=pd_mode,
        pd_prefill_nodes=int(getattr(args, "pd_prefill_nodes", 0) or 0),
        pd_decode_nodes=int(getattr(args, "pd_decode_nodes", 0) or 0),
        pd_prefill_tp=int(getattr(args, "pd_prefill_tp", 0) or 0),
        pd_decode_tp=int(getattr(args, "pd_decode_tp", 0) or 0),
        description=args.description,
        owner_id=owner_id,
        session_id=session_id,
        extra_env=env,
        extra_labels=extra_labels,
    )

    with safe_client.from_env() as safe:
        wid: str | None = None
        existing = _load_state()
        prior_wid = (existing.get("rayjob_id") or "").strip()
        prior_is_dynamo = existing.get("backend") == "dynamo"
        if prior_wid and prior_is_dynamo and not getattr(args, "recreate", False):
            try:
                prior_wl = safe.get_workload(prior_wid)
            except safe_client.SafeApiError as exc:
                if exc.status == 404:
                    info(f"prior dynamo workload {prior_wid} gone; creating fresh")
                else:
                    raise
            else:
                if str(prior_wl.get("phase") or "?") in _TERMINAL_FAIL_PHASES:
                    info(f"prior dynamo workload {prior_wid} terminal; creating fresh")
                else:
                    info(f"reusing dynamo workload {prior_wid}; resume polling")
                    wid = prior_wid

        if wid is None:
            info(f"creating DynamoDeployment (workspace={workspace} nodes={args.nodes})")
            wid = safe.create_workload(body)
            info(f"workload created: {wid}")

        # Checkpoint id immediately (idempotency: overlapping retries reuse).
        st = dict(_load_state())
        st.update({
            "backend": "dynamo",
            "rayjob_id": wid,
            "workspace": workspace,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "ssh_key_path": str(priv_key),
            "ssh_port": args.ssh_port,
            "framework": args.backend_framework,
            "pd_mode": pd_mode,
            "kv_transfer_backend": args.kv_transfer_backend,
            "service_url": dynamo_support.frontend_service_url(wid, workspace),
        })
        _save_state(st)

        workload: dict[str, Any] = {}
        if args.no_wait:
            info("--no-wait set; not polling for Running")
        else:
            def _fetch():
                wl = safe.get_workload(wid)
                return wl, f"phase={wl.get('phase', '?')}"

            workload = _short_poll(
                label=f"dynamo workload {wid}",
                fetch=_fetch,
                is_ok=lambda w: w.get("phase") in _TERMINAL_OK_PHASES,
                is_fail=lambda w: w.get("phase") in _TERMINAL_FAIL_PHASES,
                interval_s=args.poll_interval,
                timeout_s=_poll_timeout_from_args(args),
                failure_diag=_summarize_workload_failure,
                quiet_fetch_error_grace_s=_SAFE_GET_WORKLOAD_404_GRACE_S,
                is_quiet_fetch_error=_is_safe_get_workload_404,
            )

        # Resolve the frontend service URL from live service info when possible.
        service_url = st["service_url"]
        try:
            svc = safe.get_workload_service(wid)
            service_url = dynamo_support.frontend_service_url(wid, workspace, svc)
        except safe_client.SafeApiError as exc:
            warn(f"get_workload_service failed ({exc}); using conventional DNS")

        # Discover worker pod IPs from SaFE GetWorkload .pods (SaFE populates
        # DGD child pods with role-indexed resourceId; see discover_role_pods).
        roles = (
            dynamo_support.discover_role_pods(workload, pd_mode=pd_mode) if workload
            else {"frontend": [], "prefill": [], "decode": [], "worker": []}
        )
        worker_ips = [p["podIP"] for p in roles["worker"]]
        prefill_ips = [p["podIP"] for p in roles["prefill"]]
        decode_ips = [p["podIP"] for p in roles["decode"]]
        gpu_ips = (prefill_ips + decode_ips) if pd_mode == "disaggregated" else worker_ips
        if workload and not gpu_ips:
            warn(
                "discovered 0 GPU pod IPs from GetWorkload.pods; SaFE may still "
                "be syncing DGD pods. Re-run create-dynamo to refresh."
            )

    merged = dict(_load_state())
    merged.update({
        "backend": "dynamo",
        "rayjob_id": wid,
        "workspace": workspace,
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "ssh_key_path": str(priv_key),
        "ssh_port": args.ssh_port,
        "framework": args.backend_framework,
        "pd_mode": pd_mode,
        "kv_transfer_backend": args.kv_transfer_backend,
        "service_url": service_url,
        "worker_pod_ips": worker_ips,
        "prefill_pod_ips": prefill_ips,
        "decode_pod_ips": decode_ips,
        "last_create_request": {
            "image": args.image, "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node, "kind": "DynamoDeployment",
            "pd_mode": pd_mode,
        },
    })
    _save_state(merged)

    # Best-effort SSH reachability probe (non-fatal: pods may still be booting).
    if gpu_ips:
        reachable = sum(
            1 for ip in gpu_ips
            if ssh_client.probe_ssh(ip, key_path=priv_key, port=args.ssh_port)
        )
        info(f"ssh reachable GPU pods: {reachable}/{len(gpu_ips)}")

    info(f"state written to {STATE_FILE}")
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0


def _dynamo_require_state() -> dict[str, Any]:
    """Load dynamo state; require an ssh key + at least one GPU pod IP."""
    state = _load_state()
    if state.get("backend") != "dynamo":
        raise RuntimeError("state backend is not 'dynamo'; run create-dynamo first")
    has_gpu_pods = bool(
        state.get("worker_pod_ips")
        or state.get("prefill_pod_ips")
        or state.get("decode_pod_ips")
    )
    if not has_gpu_pods:
        raise RuntimeError(
            "no GPU pod IPs in state; re-run create-dynamo (LWS pods may "
            "not have had IPs yet when the workload reached Running)"
        )
    if not state.get("ssh_key_path"):
        raise RuntimeError("no ssh_key_path in state; re-run create-dynamo")
    return state


# Env-var prefixes forwarded from the controller (prompt -> setup_env.sh ->
# os.environ) to the SSH-launched framework child. These are sandbox-side tuning
# vars that are NOT in the pod container env and are NOT recovered from pid1, so
# without explicit forwarding the child sees framework defaults (e.g. mori
# SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK defaults to 4096 and prefill
# aborts when chunked_prefill_size exceeds it).
_FORWARD_ENV_PREFIXES = ("MORI_", "SGLANG_MORI_", "SGLANG_DISAGGREGATION_")


def _collect_forward_env() -> dict[str, str]:
    """Read prompt-provided tuning vars from os.environ for SSH forwarding."""
    fwd = {
        k: v
        for k, v in os.environ.items()
        if any(k.startswith(p) for p in _FORWARD_ENV_PREFIXES)
    }
    # Multi-node torch profiler: the dynamo SSH path (unlike the RayJob path in
    # launch_multinode.py) never pins SGLANG_TORCH_PROFILER_DIR, so sglang writes
    # traces to pod-local /tmp where the sandbox cannot read them -> roofline's
    # profile_no_trace_failed. Translate the controller's shared-FS trace dir
    # (HYPERLOOM_MN_PROFILE_TRACE_DIR, set by restart_server_for_round) into
    # SGLANG_TORCH_PROFILER_DIR so the SSH-launched sglang emits traces to the
    # wekafs path both server pods and the sandbox mount.
    trace_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if trace_dir and "SGLANG_TORCH_PROFILER_DIR" not in fwd:
        fwd["SGLANG_TORCH_PROFILER_DIR"] = trace_dir
    # Explicit per-variant env overrides (e.g. specialist-proposed MoE
    # tuning) come through HYPERLOOM_MN_EXTRA_FWD_ENV as a JSON object set
    # by restart_server_for_round. Unlike _FORWARD_ENV_PREFIXES these are
    # forwarded verbatim regardless of key prefix, so an explore variant's
    # ``extra_envs`` reach the SSH-launched sglang. They take precedence
    # over prefix-matched values for the same key (explicit > ambient).
    extra_fwd = os.environ.get("HYPERLOOM_MN_EXTRA_FWD_ENV", "").strip()
    if extra_fwd:
        try:
            parsed = json.loads(extra_fwd)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    fwd[str(k)] = str(v)
        except (ValueError, TypeError):
            warn(
                "HYPERLOOM_MN_EXTRA_FWD_ENV is not valid JSON; "
                "skipping per-variant env forwarding"
            )
    return fwd


def _dynamo_fanout_launch(
    state: dict[str, Any], launch_args: str, worker_ips: list[str], *,
    label: str, poll_timeout: int, print_logs: bool,
) -> tuple[int, list[dict]]:
    """Ship + run launch_dynamo_node.py on each pod in ``worker_ips`` over SSH.

    Returns ``(rc, per_pod_results)``. rc != 0 if any pod's launcher exits
    non-zero. The SAME launch_args go to every pod in the group — each
    self-determines its node-rank from $LWS_WORKER_INDEX pod-side.
    """
    script = _read_pod_script("launch_dynamo_node.py")
    key_path = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    forward_env = _collect_forward_env()
    if forward_env:
        info(f"{label}: forwarding {len(forward_env)} tuning env vars to SSH child")
    results: list[dict] = []
    rc_total = 0
    for ip in worker_ips:
        info(f"{label}: ssh -> {ip}:{port}")
        try:
            cp = ssh_client.ssh_run_script(
                ip, script, "python3", launch_args,
                key_path=key_path, port=port, timeout=poll_timeout,
                env=forward_env,
            )
        except subprocess.TimeoutExpired:
            warn(f"{label}: {ip} timed out after {poll_timeout}s")
            results.append({"podIP": ip, "rc": 124, "error": "timeout"})
            rc_total = 1
            continue
        parsed = _extract_pod_json(cp.stdout or "")
        rec = {"podIP": ip, "rc": cp.returncode, "summary": parsed}
        if cp.returncode != 0:
            rec["stderr"] = (cp.stderr or "")[-1500:]
            rc_total = 1
        results.append(rec)
        if print_logs:
            print(f"--- {ip} stdout ---\n{cp.stdout}\n--- {ip} stderr ---\n{cp.stderr}")
    return rc_total, results


def _dynamo_restart_server(args: argparse.Namespace) -> int:
    """Dynamo restart: SSH fan-out launch_dynamo_node.py to every worker pod.

    Each pod kills its prior server (PID file) and relaunches dynamo.sglang /
    dynamo.vllm wired with --nnodes/--node-rank/--dist-init-addr (rank from the
    pod's own $LWS_WORKER_INDEX). The launcher detaches immediately, so this
    returns once every rank has SPAWNED — readiness (MoE cold start) is polled
    sandbox-side against the frontend service_url, never blocked here.
    """
    state = _dynamo_require_state()
    framework = (args.framework or state.get("framework") or "sglang").lower()
    if framework not in ("sglang", "vllm"):
        raise RuntimeError(f"unsupported framework: {framework!r}")
    # The deployment topology is fixed at create time, so state.pd_mode is
    # authoritative: a PD deployment must restart in PD mode even if the
    # caller's --pd-mode defaulted to colocated/aggregated.
    pd_mode = (
        "disaggregated"
        if (getattr(args, "pd_mode", "") or "").lower() == "disaggregated"
        or state.get("pd_mode") == "disaggregated"
        else "aggregated"
    )
    kv = (
        getattr(args, "pd_transfer_backend", "") or state.get("kv_transfer_backend") or ""
    )
    poll_timeout = _poll_timeout_from_args(args)
    print_logs = getattr(args, "print_logs", False)
    rc_total = 0
    all_results: dict[str, Any] = {}

    if pd_mode == "disaggregated":
        if framework != "sglang":
            raise RuntimeError("PD disaggregation is sglang-only on the Dynamo backend")
        # Prefill group + decode group: each is its own LWS, each pod uses its
        # own $LWS_WORKER_INDEX/$LWS_LEADER_ADDRESS. We send per-group tp/nnodes
        # and the matching --disaggregation-mode.
        pn = int(getattr(args, "pd_prefill_nodes", 0) or 0) or len(state.get("prefill_pod_ips") or [])
        dn = int(getattr(args, "pd_decode_nodes", 0) or 0) or len(state.get("decode_pod_ips") or [])
        ptp = int(getattr(args, "pd_prefill_tp", 0) or 0) or int(args.tp)
        dtp = int(getattr(args, "pd_decode_tp", 0) or 0) or int(args.tp)
        # Per-role EP / extra-args (InferenceX disagg recipes differ between
        # prefill and decode). 0 / "" => fall back to the shared --ep /
        # --extra-args so legacy single-flag callers are unchanged. The
        # shared --extra-args is the common base; the per-role string is
        # appended after it (role-specific flags win on duplicate keys via
        # sglang's own last-wins argparse).
        shared_ep = int(getattr(args, "ep", 1) or 1)
        shared_extra = getattr(args, "extra_args", "") or ""
        pep = int(getattr(args, "pd_prefill_ep", 0) or 0) or shared_ep
        dep = int(getattr(args, "pd_decode_ep", 0) or 0) or shared_ep
        prefill_extra = (
            shared_extra + " " + (getattr(args, "pd_prefill_extra_args", "") or "")
        ).strip()
        decode_extra = (
            shared_extra + " " + (getattr(args, "pd_decode_extra_args", "") or "")
        ).strip()
        for role, ips, rnnodes, rtp, rep, rextra in (
            ("prefill", state.get("prefill_pod_ips") or [], pn, ptp, pep, prefill_extra),
            ("decode", state.get("decode_pod_ips") or [], dn, dtp, dep, decode_extra),
        ):
            if not ips:
                continue
            launch_args = dynamo_support.build_node_launch_args(
                framework=framework, model=args.model, tp=rtp,
                nnodes=max(1, rnnodes), ep=rep,
                extra_args=rextra,
                health_wait_sec=0, disagg_mode=role, kv_transfer_backend=kv,
            )
            info(f"dynamo restart-server PD {role}: tp={rtp} ep={rep} "
                 f"nnodes={rnnodes} pods={len(ips)} kv={kv} extra={rextra!r}")
            rc, results = _dynamo_fanout_launch(
                state, launch_args, list(ips), label=f"restart-{role}",
                poll_timeout=poll_timeout, print_logs=print_logs,
            )
            rc_total = rc_total or rc
            all_results[role] = results
    else:
        nnodes = int(state.get("nodes") or 1)
        worker_ips = state.get("worker_pod_ips") or []
        launch_args = dynamo_support.build_node_launch_args(
            framework=framework, model=args.model, tp=args.tp, nnodes=nnodes,
            ep=int(getattr(args, "ep", 1) or 1),
            extra_args=getattr(args, "extra_args", "") or "",
            health_wait_sec=0,
        )
        info(f"dynamo restart-server: framework={framework} model={args.model} "
             f"tp={args.tp} nnodes={nnodes} workers={len(worker_ips)}")
        rc_total, results = _dynamo_fanout_launch(
            state, launch_args, list(worker_ips), label="restart",
            poll_timeout=poll_timeout, print_logs=print_logs,
        )
        all_results["worker"] = results

    state["last_restart_framework"] = framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = int(args.tp)
    state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
    state["last_restart_pd_mode"] = pd_mode
    state["last_restart_extra_args"] = _normalize_extra_args(
        getattr(args, "extra_args", "")
    )
    if pd_mode == "disaggregated":
        # Persist per-role knobs so a state-only resume (env lost on sandbox
        # recreate) reproduces the same prefill/decode topology.
        state["last_restart_pd_prefill_nodes"] = int(getattr(args, "pd_prefill_nodes", 0) or 0)
        state["last_restart_pd_decode_nodes"] = int(getattr(args, "pd_decode_nodes", 0) or 0)
        state["last_restart_pd_prefill_tp"] = int(getattr(args, "pd_prefill_tp", 0) or 0)
        state["last_restart_pd_decode_tp"] = int(getattr(args, "pd_decode_tp", 0) or 0)
        state["last_restart_pd_prefill_ep"] = int(getattr(args, "pd_prefill_ep", 0) or 0)
        state["last_restart_pd_decode_ep"] = int(getattr(args, "pd_decode_ep", 0) or 0)
        state["last_restart_pd_prefill_extra_args"] = getattr(args, "pd_prefill_extra_args", "") or ""
        state["last_restart_pd_decode_extra_args"] = getattr(args, "pd_decode_extra_args", "") or ""
    state["last_restart_results"] = all_results
    _save_state(state)
    print(json.dumps(
        {"backend": "dynamo", "pd_mode": pd_mode, "rc": rc_total, "results": all_results},
        indent=2,
    ))
    if rc_total != 0:
        info("dynamo restart: at least one launcher failed; see results")
        return 1
    info("dynamo servers launched; benchmark via $service_url (frontend :8000)")
    return 0


def _dynamo_all_gpu_ips(state: dict[str, Any]) -> list[str]:
    """Every GPU pod IP to act on: PD => prefill+decode, else worker."""
    pd = (state.get("pd_mode") or "aggregated").lower()
    if pd == "disaggregated":
        return list(state.get("prefill_pod_ips") or []) + list(state.get("decode_pod_ips") or [])
    return list(state.get("worker_pod_ips") or [])


def _dynamo_kill_inference(args: argparse.Namespace) -> int:
    """Dynamo kill: SSH fan-out launch_dynamo_node.py --kill-only to every GPU pod."""
    state = _dynamo_require_state()
    framework = (
        state.get("last_restart_framework") or state.get("framework") or "sglang"
    ).lower()
    gpu_ips = _dynamo_all_gpu_ips(state)
    launch_args = dynamo_support.build_node_launch_args(
        framework=framework, model="", tp=0,
        nnodes=int(state.get("nodes") or 1), kill_only=True,
    )
    info(f"dynamo kill-inference: framework={framework} pods={len(gpu_ips)}")
    rc, results = _dynamo_fanout_launch(
        state, launch_args, gpu_ips, label="kill",
        poll_timeout=_poll_timeout_from_args(args),
        print_logs=getattr(args, "print_logs", False),
    )
    state["last_kill_results"] = results
    _save_state(state)
    print(json.dumps(
        {"backend": "dynamo", "action": "kill", "rc": rc, "results": results},
        indent=2,
    ))
    return 0 if rc == 0 else 1


def _dynamo_ssh_node_op(
    state: dict[str, Any], ip: str, op_args: str, *, timeout: int,
) -> tuple[dict | None, dict]:
    """Ship kernel_node_ops.py to one pod over SSH and run one subcommand.

    Returns ``(parsed_json_or_None, transport)`` where transport carries the
    ssh rc / stderr for diagnostics.
    """
    script = _read_pod_script("kernel_node_ops.py")
    key = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    try:
        cp = ssh_client.ssh_run_script(
            ip, script, "python3", op_args, key_path=key, port=port, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, {"rc": 124, "stderr": f"timeout after {timeout}s"}
    return _extract_pod_json(cp.stdout or ""), {
        "rc": cp.returncode, "stderr": (cp.stderr or "")[-1500:],
    }


def _dynamo_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Dynamo apply-tracelens-patch: SSH fan-out the TraceLens SGLang patch
    set to every GPU pod via ``apply_tracelens_patch_multinode.py --local``.

    The ray path submits a Ray-actor fan-out; the dynamo path has no ray, so
    each GPU pod runs the patcher locally over SSH. Both annotate the sglang
    torch.profiler output the same way, so the NFS-shared trace dir (see
    SGLANG_TORCH_PROFILER_DIR forwarding in _collect_forward_env) is consumable
    by TraceLens identically — only the dispatch differs. Idempotent: the
    in-pod script sentinel-greps and returns status=skipped on already-patched
    pods, so it is safe to call on every restart_server_for_round.
    """
    state = _dynamo_require_state()
    tracelens_root = (
        args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    )
    if not tracelens_root:
        err(
            "apply-tracelens-patch (dynamo) requires --tracelens-root or "
            "$TRACELENS_ROOT (an NFS path visible from every GPU pod)"
        )
        return EXIT_CONFIG_ERROR
    gpu_ips = _dynamo_all_gpu_ips(state)
    if not gpu_ips:
        err("apply-tracelens-patch (dynamo): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    script = _read_pod_script("apply_tracelens_patch_multinode.py")
    key_path = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    pin = getattr(args, "sglang_version_pin", None) or ""
    op_args = f"--local --tracelens-root {tracelens_root!r}"
    if pin:
        op_args += f" --sglang-version-pin {pin!r}"
    timeout = _poll_timeout_from_args(args)
    per_pod: list[dict] = []
    failures: list[dict] = []
    # Pod-side interpreter: sglang lives in /opt/venv on the canonical
    # ROCm sglang-dynamo images; /usr/bin/python3 lacks sglang so
    # _apply_on_pod's `import sglang` fails with "No module named 'sglang'".
    # Allow override via $HYPERLOOM_MN_POD_PYTHON.
    pod_python = os.environ.get("HYPERLOOM_MN_POD_PYTHON", "/opt/venv/bin/python")
    for ip in gpu_ips:
        info(f"apply-tracelens-patch (dynamo): ssh -> {ip}")
        try:
            cp = ssh_client.ssh_run_script(
                ip, script, pod_python, op_args,
                key_path=key_path, port=port, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append({"host": ip, "error": f"timeout after {timeout}s"})
            continue
        parsed = _extract_pod_json(cp.stdout or "")
        pods = (parsed or {}).get("per_pod") or []
        if parsed and str(parsed.get("status")) in ("applied", "skipped") and pods:
            for r in pods:
                r["host"] = ip
                per_pod.append(r)
        else:
            failures.append({
                "host": ip,
                "error": (parsed or {}).get("error")
                or (cp.stderr or "")[-800:] or "unknown",
                "rc": cp.returncode,
            })
    overall = "applied" if not failures else "failed"
    if overall == "applied" and per_pod and all(
        r.get("status") == "skipped" for r in per_pod
    ):
        overall = "skipped"
    print(json.dumps(
        {"command": "apply-tracelens-patch", "backend": "dynamo",
         "status": overall, "per_pod": per_pod, "failures": failures},
        indent=2, sort_keys=True,
    ))
    return 0 if not failures else 1


def _dynamo_apply_patch(args: argparse.Namespace) -> int:
    """Dynamo apply-patch: SSH fan-out kernel_node_ops.py apply to every GPU pod.

    Emits the SAME JSON shape as kernel_patch_multinode.py (command/status/
    per_node/failures), with ``per_node[].host`` keyed by the pod IP so the
    sandbox builds an IP->backup_path revert map.
    """
    state = _dynamo_require_state()
    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_path.read_bytes()).decode("ascii")
    gpu_ips = _dynamo_all_gpu_ips(state)
    op_args = (
        f"apply --target-path {args.target_path!r} --patch-b64 {patch_b64!r} "
        f"--backup-dir {args.backup_dir!r} --kernel-id {args.kernel_id!r}"
    )
    per_node: list[dict] = []
    failures: list[dict] = []
    for ip in gpu_ips:
        info(f"apply-patch (dynamo): ssh -> {ip}")
        parsed, tx = _dynamo_ssh_node_op(state, ip, op_args, timeout=args.timeout_sec)
        if parsed and str(parsed.get("status")) == "ok":
            # Override host with the pod IP so revert targets the same pod.
            parsed["host"] = ip
            per_node.append(parsed)
        else:
            failures.append({"host": ip, "error": (parsed or {}).get("error")
                             or tx.get("stderr") or "unknown", **tx})
    payload = {
        "command": "apply", "target_path": args.target_path,
        "kernel_id": args.kernel_id, "backup_dir": args.backup_dir,
        "per_node": per_node, "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _dynamo_revert_patch(args: argparse.Namespace) -> int:
    """Dynamo revert-patch: SSH each pod in the IP->backup_path map + restore."""
    state = _dynamo_require_state()
    try:
        backup_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    if not backup_map:
        err("--backup-map-json must be a non-empty {host: backup_path} object")
        return EXIT_CONFIG_ERROR
    per_node: list[dict] = []
    failures: list[dict] = []
    for ip, backup_path in backup_map.items():
        info(f"revert-patch (dynamo): ssh -> {ip}")
        op_args = (
            f"revert --target-path {args.target_path!r} --backup-path {backup_path!r}"
        )
        parsed, tx = _dynamo_ssh_node_op(state, ip, op_args, timeout=args.timeout_sec)
        if parsed and str(parsed.get("status")) in ("restored", "noop_missing_backup"):
            per_node.append({"host": ip, **parsed})
        else:
            failures.append({"host": ip, "error": (parsed or {}).get("error")
                             or tx.get("stderr") or "unknown", **tx})
    payload = {
        "command": "revert", "target_path": args.target_path,
        "per_node": per_node, "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _dynamo_kernel_bench(args: argparse.Namespace) -> int:
    """Dynamo kernel-bench: run the micro-benchmark on ONE GPU pod over SSH."""
    state = _dynamo_require_state()
    gpu_ips = _dynamo_all_gpu_ips(state)
    if not gpu_ips:
        err("kernel-bench (dynamo): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    ip = gpu_ips[0]
    if args.files_b64_json:
        try:
            json.loads(args.files_b64_json)
        except json.JSONDecodeError as exc:
            err(f"--files-b64-json not valid JSON: {exc}")
            return EXIT_CONFIG_ERROR
    op_args = (
        f"bench --workspace {args.workspace!r} "
        f"--bench-command {args.bench_command!r} "
        f"--files-b64-json {(args.files_b64_json or '{}')!r} "
        f"--result-glob {args.result_glob!r} --timeout-sec {int(args.timeout_sec)}"
    )
    info(f"kernel-bench (dynamo): ssh -> {ip}")
    parsed, tx = _dynamo_ssh_node_op(
        state, ip, op_args, timeout=args.timeout_sec + 60,
    )
    if parsed is None:
        err(f"kernel-bench (dynamo): no JSON from pod (ssh rc={tx.get('rc')})")
        if getattr(args, "print_logs", False):
            print(tx.get("stderr", ""))
        return EXIT_TRANSIENT
    payload = {"command": "bench",
               "status": "ok" if str(parsed.get("status")) == "ok" else "failed",
               "result": parsed}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


def _resolve_geak_src(explicit: str | None) -> str:
    """Resolve the shared-FS GEAK source dir the sandbox install.sh cloned.

    Resolution: --geak-src > $HYPERLOOM_GEAK_SRC > $HYPERLOOM_ROOT/geak >
    $USER_DATA_PATH/runtime/geak. Must be a path both sandbox and pod see
    (under $USER_DATA_PATH).
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("HYPERLOOM_GEAK_SRC", "").strip()
    if env:
        return env
    root = os.environ.get("HYPERLOOM_ROOT", "").strip()
    if root:
        return f"{root.rstrip('/')}/geak"
    udp = os.environ.get("USER_DATA_PATH", "").strip()
    if udp:
        return f"{udp.rstrip('/')}/runtime/geak"
    return ""


def cmd_install_geak(args: argparse.Namespace) -> int:
    """Install the GEAK CLI on every Dynamo GPU pod over SSH (idempotent).

    pip-installs the shared-FS GEAK checkout into each pod's framework venv so
    the kernel-agent SSH placement (run_geak_over_ssh) finds ``geak`` on PATH.
    Dynamo-only (kernel-agent on RayJob uses the Ray runtime, not this).
    """
    state = _dynamo_require_state()
    geak_src = _resolve_geak_src(getattr(args, "geak_src", None))
    if not geak_src:
        err("install-geak: cannot resolve GEAK source dir; pass --geak-src or "
            "set $HYPERLOOM_ROOT / $USER_DATA_PATH")
        return EXIT_CONFIG_ERROR
    script = _read_pod_script("install_geak_node.sh")
    key = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    gpu_ips = _dynamo_all_gpu_ips(state)
    info(f"install-geak (dynamo): geak_src={geak_src} pods={len(gpu_ips)}")
    results: list[dict] = []
    rc_total = 0
    for ip in gpu_ips:
        info(f"install-geak: ssh -> {ip}")
        try:
            cp = ssh_client.ssh_run_script(
                ip, script, "bash", f"{geak_src!r}",
                key_path=key, port=port, timeout=_poll_timeout_from_args(args),
            )
        except subprocess.TimeoutExpired:
            results.append({"host": ip, "status": "failed", "reason": "timeout"})
            rc_total = 1
            continue
        parsed = _extract_pod_json(cp.stdout or "") or {
            "status": "failed", "reason": (cp.stderr or "")[-500:],
        }
        parsed["host"] = ip
        results.append(parsed)
        if str(parsed.get("status")) not in ("installed", "skipped"):
            rc_total = 1
        if getattr(args, "print_logs", False):
            print(f"--- {ip} ---\n{cp.stdout}\n{cp.stderr}")
    print(json.dumps({"command": "install-geak", "results": results,
                      "status": "ok" if rc_total == 0 else "partial"}, indent=2))
    return rc_total


def cmd_install_oob(args: argparse.Namespace) -> int:
    """Install the OOB backend (oob/claude/codex/@cursor) on every Dynamo GPU
    pod over SSH (idempotent). Mirrors install.sh ensure_node + ensure_oob so
    claude/codex/cursor kernel-agent backends work on the Dynamo backend.

    OOB python CLI installs from the shared-NFS ``$OOB_SRC`` checkout; the npm
    CLIs (claude/codex/@cursor-sdk) install per pod. Credentials are forwarded
    over SSH stdin (never argv). Dynamo-only.
    """
    state = _dynamo_require_state()
    oob_src = (getattr(args, "oob_src", None) or os.environ.get("OOB_SRC", "")).strip()
    if not oob_src:
        err("install-oob: no OOB source; pass --oob-src or set $OOB_SRC")
        return EXIT_CONFIG_ERROR
    script = _read_pod_script("install_oob_node.sh")
    env: dict[str, str] = {"OOB_SRC": oob_src}
    for k in ("OOB_BASE_URL", "OOB_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        v = os.environ.get(k, "").strip()
        if v:
            env[k] = v
    # OOB_API_KEY falls back to SAFE_API_KEY (ADDENDUM-13 credential fanout).
    if "OOB_API_KEY" not in env:
        sk = os.environ.get("SAFE_API_KEY", "").strip()
        if sk:
            env["OOB_API_KEY"] = sk
    key = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    gpu_ips = _dynamo_all_gpu_ips(state)
    # node+npm+pip installs are slow; allow a generous timeout (npm registry).
    timeout = max(_poll_timeout_from_args(args), 1800)
    info(f"install-oob (dynamo): oob_src={oob_src} pods={len(gpu_ips)}")
    results: list[dict] = []
    rc_total = 0
    for ip in gpu_ips:
        info(f"install-oob: ssh -> {ip}")
        try:
            cp = ssh_client.ssh_run_bash_with_env(
                ip, script, env, key_path=key, port=port, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            results.append({"host": ip, "status": "failed", "reason": "timeout"})
            rc_total = 1
            continue
        parsed = _extract_pod_json(cp.stdout or "") or {
            "status": "failed", "reason": (cp.stderr or "")[-500:],
        }
        parsed["host"] = ip
        results.append(parsed)
        if str(parsed.get("status")) not in ("installed", "skipped"):
            rc_total = 1
        if getattr(args, "print_logs", False):
            print(f"--- {ip} ---\n{cp.stdout}\n{cp.stderr}")
    print(json.dumps({"command": "install-oob", "results": results,
                      "status": "ok" if rc_total == 0 else "partial"}, indent=2))
    return rc_total


def install_geak_on_pods_best_effort() -> int:
    """Best-effort GEAK install on the Dynamo GPU pods (provisioner hook).

    No-op (returns 0) for non-dynamo state. Failures are logged but do not
    abort provisioning — the kernel phase will surface a clear pod-side
    ``geak CLI not found`` error if install genuinely failed.
    """
    if _load_state().get("backend") != "dynamo":
        return 0
    ns = argparse.Namespace(
        geak_src=None, print_logs=False,
        poll_interval=_DEFAULT_POLL_INTERVAL_S,
        poll_timeout=_resolve_poll_timeout_s(),
    )
    try:
        return cmd_install_geak(ns)
    except Exception as exc:  # noqa: BLE001
        warn(f"install-geak skipped: {type(exc).__name__}: {exc}")
        return 0


def install_oob_on_pods_best_effort() -> int:
    """Best-effort OOB install on the Dynamo GPU pods (provisioner hook).

    No-op (0) for non-dynamo. Failures are logged but never abort provisioning.
    """
    if _load_state().get("backend") != "dynamo":
        return 0
    ns = argparse.Namespace(
        oob_src=None, print_logs=False,
        poll_interval=_DEFAULT_POLL_INTERVAL_S,
        poll_timeout=_resolve_poll_timeout_s(),
    )
    try:
        return cmd_install_oob(ns)
    except Exception as exc:  # noqa: BLE001
        warn(f"install-oob skipped: {type(exc).__name__}: {exc}")
        return 0


def install_kernel_tools_on_pods_best_effort() -> int:
    """Provisioner hook: install BOTH GEAK and OOB on the Dynamo GPU pods.

    No-op for non-dynamo. Returns non-zero only if a sub-install reported a
    hard failure (best-effort; provisioning continues regardless).
    """
    rc_geak = install_geak_on_pods_best_effort()
    rc_oob = install_oob_on_pods_best_effort()
    return rc_geak or rc_oob


# ---------------------------------------------------------------------------
# Subcommand: bootstrap


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Run the BYOI bootstrap script inside the RayJob via Ray Dashboard REST.

    The bootstrap script lives in ``multi_node/scripts/bootstrap.sh``
    inside this Hyperloom checkout (i.e. visible from the SANDBOX, not
    the RayJob pod). We stream its contents into the head pod via a
    heredoc-style entrypoint so the pod doesn't need filesystem access
    to the sandbox. Pass ``--script PATH`` to override with a script
    that's already pod-visible (e.g. baked into the image).
    """
    state = _require_state("rayjob_id", "head_pod_ip")
    head_ip = state["head_pod_ip"]

    if args.script:
        # Operator override: assume the path is pod-visible.
        entrypoint = f"bash {args.script}" + (" --force" if args.force else "")
    else:
        bootstrap_sh = _read_pod_script("bootstrap.sh")
        force_arg = " --force" if args.force else ""
        entrypoint = (
            "set -euo pipefail; "
            "WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p \"$WORK_DIR\"; "
            "cat > \"$WORK_DIR/bootstrap.sh\" <<'__MN_BOOT_EOF__'\n"
            f"{bootstrap_sh}__MN_BOOT_EOF__\n"
            f"chmod +x \"$WORK_DIR/bootstrap.sh\"; "
            f"\"$WORK_DIR/bootstrap.sh\"{force_arg}"
        )

    with ray_dashboard.RayDashboardClient(head_ip) as ray:
        info(f"submitting bootstrap entrypoint: {entrypoint}")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id}")

        def _fetch():
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"bootstrap {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        info(f"bootstrap done: {result.get('status')}")
        if args.print_logs:
            logs = ray.get_job_logs(sub_id)
            print(logs)

    # Persist the bootstrap submission id for later debugging.
    state["last_bootstrap_submission_id"] = sub_id
    _save_state(state)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: verify


def cmd_verify(args: argparse.Namespace) -> int:
    """Sanity-check the toolchain bootstrap installed inside the RayJob."""
    state = _require_state("head_pod_ip")
    head_ip = state["head_pod_ip"]

    # Source env file (PATH → /opt/venv), then verify ``ray`` on PATH.
    # ``oob`` / ``claude`` / ``codex`` excluded: head pod never invokes
    # these CLIs (see bootstrap.sh "# --- 2. / 2c. (removed)" comments).
    script = (
        "set -e; "
        "if [ -f /etc/profile.d/hyperloom-env.sh ]; "
        "then source /etc/profile.d/hyperloom-env.sh; fi; "
        "for bin in ray; do "
        "  echo \"-- which $bin --\"; "
        "  which \"$bin\" || { echo \"MISSING: $bin\" >&2; exit 1; }; "
        "done; "
        "echo OK"
    )
    entrypoint = script  # _wrap_for_dash will wrap as bash; -lc breaks PATH
    with ray_dashboard.RayDashboardClient(head_ip) as ray:
        info("submitting verify entrypoint")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id}")

        def _fetch():
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"verify {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs or str(result.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES:
            logs = ray.get_job_logs(sub_id)
            print(logs)
        info(f"verify done: {result.get('status')}")

    state["last_verify_submission_id"] = sub_id
    _save_state(state)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: restart-server


_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_pod_script(name: str) -> str:
    """Read a pod-side bash script from ``multi_node/scripts/`` and return
    its contents. The CLI embeds it into the Ray Dashboard ``/api/jobs/``
    entrypoint at submit time so the head pod doesn't need filesystem
    access to the sandbox.

    Splitting the bash into separate files (vs inlining strings) keeps
    each one independently readable / editable / oob-optimizable, which
    is what the ``scripts_subset=all_three`` decision optimizes for.
    """
    p = _SCRIPTS_DIR / name
    if not p.is_file():
        raise RuntimeError(
            f"missing pod-side script: {p}. "
            "Did you trim the multi_node/scripts/ directory?"
        )
    return p.read_text(encoding="utf-8")


def _build_restart_entrypoint(
    args: argparse.Namespace,
    pid_file: str,
    log_file: str,
) -> str:
    """Compose the Ray Dashboard entrypoint for one restart cycle.

    Strategy: stream the kill_server.sh + launch_server.sh contents into
    the head pod via heredoc, then invoke them with the right args.
    The pod doesn't need to know about hyperloom on disk; everything is
    self-contained in the entrypoint.

    Strict IR-5 (no ``pkill -f sglang``): kill_server.sh uses the PID file
    that launch_server.sh wrote in the previous cycle.
    """
    framework = args.framework.lower()
    if framework not in ("sglang", "vllm"):
        raise RuntimeError(f"unsupported framework: {args.framework!r} (use sglang or vllm)")

    kill_sh = _read_pod_script("kill_server.sh")
    launch_sh = _read_pod_script("launch_server.sh")

    wait_flag = "--no-wait-health" if args.no_wait_health else "--wait-health"

    # Use stable heredoc terminators that won't collide with the script
    # bodies. We also chmod +x so the spawned shells can exec them.
    entrypoint = (
        "set -euo pipefail; "
        f"WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p \"$WORK_DIR\"; "
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        f"cat > \"$WORK_DIR/launch_server.sh\" <<'__MN_LAUNCH_EOF__'\n"
        f"{launch_sh}__MN_LAUNCH_EOF__\n"
        "chmod +x \"$WORK_DIR/kill_server.sh\" \"$WORK_DIR/launch_server.sh\"; "
        f"\"$WORK_DIR/kill_server.sh\" {pid_file!s}; "
        f"\"$WORK_DIR/launch_server.sh\" {framework!s} {args.model!s} {args.tp!s} "
        f"{pid_file!s} {log_file!s} {wait_flag} -- {args.extra_args}"
    )
    return entrypoint


# Common preamble for any multi-node entrypoint. Sources the bootstrap-
# rendered env file so PATH points at /opt/venv/bin (where sglang/vllm/ray
# Python packages live in the framework image), so the subsequent
# ``python3 ...`` resolves to the venv interpreter and ``import sglang``
# / ``import ray`` work. The env file is created by bootstrap.sh; if the
# agent skipped bootstrap the source is a no-op and we fall back to the
# image's default PATH (which usually still has /opt/venv/bin via the
# image's ENTRYPOINT, but is not guaranteed). See ADDENDUM-13.
_MN_ENTRYPOINT_PREAMBLE = (
    "set -euo pipefail; "
    "if [ -f /etc/profile.d/hyperloom-env.sh ]; then "
    "source /etc/profile.d/hyperloom-env.sh; "
    "fi; "
    "WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p \"$WORK_DIR\"; "
)


def _build_kill_single_entrypoint(pid_file: str) -> str:
    """Head-pod entrypoint that only runs kill_server.sh (IR-5 PID-file kill)."""
    kill_sh = _read_pod_script("kill_server.sh")
    return (
        "set -euo pipefail; "
        f"WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p \"$WORK_DIR\"; "
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        "chmod +x \"$WORK_DIR/kill_server.sh\"; "
        f"\"$WORK_DIR/kill_server.sh\" {pid_file!s}"
    )


def _exec_kill_submission(
    head_ip: str,
    entrypoint: str,
    *,
    label: str,
    args: argparse.Namespace,
) -> str:
    """Submit a kill entrypoint via Ray Dashboard and poll to SUCCEEDED. Returns submission_id."""
    with ray_dashboard.RayDashboardClient(head_ip) as ray:
        kill_sub = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"{label} submission_id={kill_sub}")

        def _fetch_kill():
            j = ray.get_job(kill_sub)
            return j, f"kill status={j.get('status', '?')}"

        _short_poll(
            label=f"{label} {kill_sub}",
            fetch=_fetch_kill,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if getattr(args, "print_logs", False):
            logs = ray.get_job_logs(kill_sub)
            print(logs)
    return kill_sub


def _build_multinode_kill_entrypoint(pid_dir: str, grace_sec: int = 5) -> str:
    """Compose the Ray Dashboard entrypoint that kills every multi-node
    rank's server via kill_multinode.py (heredoc-embedded).

    Sends 1 entrypoint to the head pod; kill_multinode.py uses ray
    actors to fan out kills to every worker pod's PID files.
    """
    py = _read_pod_script("kill_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kill_multinode.py\" <<'__MN_KILL_PY_EOF__'\n"
        f"{py}__MN_KILL_PY_EOF__\n"
        f"python3 \"$WORK_DIR/kill_multinode.py\" "
        f"--pid-dir {pid_dir!s} --grace-sec {grace_sec}"
    )


def _extract_launcher_summary(launch_logs: str) -> dict:
    """Parse the JSON summary that launch_multinode.py writes to stdout.

    The script emits a single ``json.dumps(summary, indent=2)`` block.
    Job logs from the dashboard interleave stderr (timestamped, prefixed
    ``[launch_multinode ...]``) with stdout. We isolate the JSON by
    finding the last opening ``{`` whose closing ``}`` reaches end-of-
    text after stripping trailing whitespace, and try to ``json.loads``
    that span. Returns ``{}`` on any failure (caller treats missing
    fields as fatal).
    """
    if not launch_logs:
        return {}
    text = launch_logs.rstrip()
    # Walk backwards: find the last balanced {...} block.
    depth = 0
    end_idx = -1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end_idx = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    candidate = text[i:end_idx + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    # Continue walking; an inner brace might be a Python
                    # repr inside a stderr line, not real JSON.
                    pass
                end_idx = -1
    return {}


def _build_multinode_launch_entrypoint(
    args: argparse.Namespace,
    nnodes: int,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the Ray Dashboard entrypoint that spawns one rank per
    node via launch_multinode.py (heredoc-embedded).

    Single entrypoint to head pod; the in-pod ray driver pins one
    actor per node and starts the framework launcher with the right
    --nnodes / --node-rank / --dist-init-addr.

    Note: ranks killed by ``_build_multinode_kill_entrypoint`` MUST be
    cleared BEFORE this entrypoint runs (sequenced by cmd_restart_server),
    otherwise rank 0's old process holds :8888 and the new process
    fails to bind.
    """
    py = _read_pod_script("launch_multinode.py")
    wait_flag = "--no-wait-health" if args.no_wait_health else ""
    extra_args = args.extra_args or ""
    # Multi-node only: pin SGLANG_TORCH_PROFILER_DIR to a shared-FS path
    # that both server pods and the sandbox can read. Resolution
    # (first-match wins):
    #   1. $HYPERLOOM_MN_PROFILE_TRACE_DIR env — set by
    #      ``inference_optimizer.cli._provision_multi_node_rayjob_stack``
    #      when ``optimize`` provisions the RayJob in-process.
    #   2. Derive from state.json's ``rayjob_id`` —
    #      ``<mn_profile_trace_root>/<rayjob>/torch_trace`` where the
    #      root is anchored on ``$USER_DATA_PATH`` (see
    #      ``inference_optimizer.paths.mn_profile_trace_root``).
    #      Triggered when the agent calls ``multi_node create-rayjob``
    #      + ``restart-server`` directly (without going through
    #      ``inference_optimizer.cli._run_optimize``); in that path the
    #      env never gets set, so we recompute from the persisted rayjob_id.
    # Empty string => skip the flag entirely (single-node default).
    profiler_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if not profiler_dir:
        _st = _load_state()
        _rid = str(_st.get("rayjob_id") or "").strip()
        if _rid:
            _profiler_path = mn_profile_trace_root() / _rid / "torch_trace"
            # Ensure the shared trace dir exists before the pod-side
            # sglang/vllm process tries to write into it. Best-effort:
            # asymmetric mounts (sandbox read-only, pod read-write) may
            # PermissionError here yet still work pod-side, so we WARN
            # and continue rather than aborting the restart.
            try:
                _profiler_path.mkdir(parents=True, exist_ok=True)
            except OSError as _exc:
                warn(
                    f"cannot mkdir profile-traces dir {_profiler_path}: {_exc}; "
                    f"pod-side launch will retry the mkdir"
                )
            profiler_dir = str(_profiler_path)
            info(f"profile-traces dir derived from rayjob_id: {profiler_dir}")
    profiler_arg = f"--torch-profiler-dir {profiler_dir!r} " if profiler_dir else ""
    # Expert parallel size. ep <= 1 => no flag (sglang/vllm legacy
    # TP-shard experts); ep > 1 => launch_multinode.py translates to
    # `--enable-ep-moe --ep-size N` (sglang) or `--enable-expert-parallel`
    # (vllm). Source: cmd_restart_server's --ep argparse, defaulting
    # to 1 when not provided to keep single-node restarts unchanged.
    try:
        ep_val = int(getattr(args, "ep", 1) or 1)
    except (TypeError, ValueError):
        ep_val = 1
    ep_arg = f"--ep {ep_val} " if ep_val > 1 else ""
    # PD disaggregation: forward all PD knobs to launch_multinode.py.
    # When pd_mode != "disaggregated" we emit no PD flags so the
    # in-pod script keeps its colocated default.
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    pd_args = ""
    if pd_mode == "disaggregated":
        pn = int(getattr(args, "pd_prefill_nodes", 0) or 0)
        dn = int(getattr(args, "pd_decode_nodes", 0) or 0)
        ptp = int(getattr(args, "pd_prefill_tp", 0) or 0)
        dtp = int(getattr(args, "pd_decode_tp", 0) or 0)
        tb = (getattr(args, "pd_transfer_backend", "") or "").strip()
        ib = (getattr(args, "pd_ib_device", "") or "").strip()
        bp = int(getattr(args, "pd_bootstrap_port", 0) or 0)
        chunks = [
            "--pd-mode disaggregated",
            f"--pd-prefill-nodes {pn}",
            f"--pd-decode-nodes {dn}",
        ]
        if ptp > 0:
            chunks.append(f"--pd-prefill-tp {ptp}")
        if dtp > 0:
            chunks.append(f"--pd-decode-tp {dtp}")
        if tb:
            chunks.append(f"--pd-transfer-backend {tb}")
        if ib:
            chunks.append(f"--pd-ib-device {ib}")
        if bp > 0:
            chunks.append(f"--pd-bootstrap-port {bp}")
        pd_args = " ".join(chunks) + " "
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_multinode.py\" <<'__MN_LAUNCH_PY_EOF__'\n"
        f"{py}__MN_LAUNCH_PY_EOF__\n"
        f"python3 \"$WORK_DIR/launch_multinode.py\" "
        f"--framework {args.framework!s} --model {args.model!s} "
        f"--tp {args.tp!s} --nnodes {nnodes!s} "
        f"--pid-dir {pid_dir!s} --log-dir {log_dir!s} "
        f"{ep_arg}{profiler_arg}{pd_args}{wait_flag} --extra-args {extra_args!r}"
    )


def _build_multinode_router_entrypoint(
    args: argparse.Namespace,
    prefill_url: str,
    decode_url: str,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the Ray Dashboard entrypoint that detaches the PD router
    on the head pod via launch_router.py (heredoc-embedded).

    Distinct from the launcher entrypoint because the router runs on
    rank 0 only, has no ray.actors, and binds the public 8888 port.
    Embedding the script lets us update ``launch_router.py`` in this
    repo and have changes take effect on the next restart-server
    without rebuilding the RayJob image.
    """
    py = _read_pod_script("launch_router.py")
    public_port = 8888
    pid_file = f"{pid_dir.rstrip('/')}/router.pid"
    log_file = f"{log_dir.rstrip('/')}/router.log"
    vllm_router_cmd = (getattr(args, "pd_vllm_router_cmd", "") or "").strip()
    vrc_arg = (
        f"--vllm-router-cmd {vllm_router_cmd!r} " if vllm_router_cmd else ""
    )
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_router.py\" <<'__MN_ROUTER_PY_EOF__'\n"
        f"{py}__MN_ROUTER_PY_EOF__\n"
        f"python3 \"$WORK_DIR/launch_router.py\" "
        f"--framework {args.framework!s} "
        f"--prefill-url {prefill_url!r} --decode-url {decode_url!r} "
        f"--public-port {public_port} "
        f"--pid-file {pid_file!r} --log-file {log_file!r} "
        f"{vrc_arg}"
    )


def _build_multinode_apply_patch_entrypoint(
    target_path: str,
    patch_b64: str,
    backup_dir: str,
    kernel_id: str,
    timeout_sec: int,
) -> str:
    """Compose the Ray Dashboard entrypoint that fans out a kernel patch
    to every pod via kernel_patch_multinode.py (heredoc-embedded).

    Sends 1 entrypoint to the head pod; the in-pod script enumerates
    Ray nodes and spawns per-node actors that write the same patch to
    target_path on each pod (head + workers). See module
    kernel_patch_multinode.py for the algorithm.

    Why heredoc embedding: same reason as launch_multinode.py — keeps
    the pod-side script versioned in this repo and updatable without
    rebuilding the RayJob image.
    """
    py = _read_pod_script("kernel_patch_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kernel_patch_multinode.py\" "
        f"<<'__MN_KPATCH_PY_EOF__'\n"
        f"{py}__MN_KPATCH_PY_EOF__\n"
        f"python3 \"$WORK_DIR/kernel_patch_multinode.py\" apply "
        f"--target-path {target_path!r} "
        f"--patch-b64 {patch_b64!r} "
        f"--backup-dir {backup_dir!r} "
        f"--kernel-id {kernel_id!r} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_revert_patch_entrypoint(
    target_path: str,
    backup_map_json: str,
    timeout_sec: int,
) -> str:
    """Compose the Ray Dashboard entrypoint that fans out a revert call
    to every pod via kernel_patch_multinode.py revert (heredoc-embedded).

    ``backup_map_json`` is the per-host map of backup file paths that
    was returned by the matching ``apply`` call; callers MUST persist
    it (we recommend the kernel-agent manifest) so revert can reach
    the same pods even after a sandbox restart.
    """
    py = _read_pod_script("kernel_patch_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kernel_patch_multinode.py\" "
        f"<<'__MN_KPATCH_PY_EOF__'\n"
        f"{py}__MN_KPATCH_PY_EOF__\n"
        f"python3 \"$WORK_DIR/kernel_patch_multinode.py\" revert "
        f"--target-path {target_path!r} "
        f"--backup-map-json {backup_map_json!r} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_apply_tracelens_patch_entrypoint(
    tracelens_root: str,
    sglang_version_pin: str,
) -> str:
    """Compose the Ray Dashboard entrypoint that fans out the TraceLens
    SGLang patch set to every pod via apply_tracelens_patch_multinode.py
    (heredoc-embedded).

    Unlike apply-patch (which carries the patch payload base64-encoded in
    the entrypoint), this one only forwards ``$TRACELENS_ROOT`` (public
    TraceLens checkout) because the patches live on a wekafs path mounted
    on every pod and the in-pod script reads them directly. Avoids
    inflating the entrypoint by ~50KB of patch bytes per restart.

    The in-pod script is idempotent (sentinel grep before applying), so
    the controller can call this on every ``restart_server_for_round``
    without worrying about double-patching.
    """
    py = _read_pod_script("apply_tracelens_patch_multinode.py")
    pin_arg = ""
    if sglang_version_pin:
        pin_arg = f" --sglang-version-pin {sglang_version_pin!r}"
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/apply_tracelens_patch_multinode.py\" "
        f"<<'__MN_TLPATCH_PY_EOF__'\n"
        f"{py}__MN_TLPATCH_PY_EOF__\n"
        f"python3 \"$WORK_DIR/apply_tracelens_patch_multinode.py\" "
        f"--tracelens-root {tracelens_root!r}"
        f"{pin_arg}"
    )


def _build_multinode_kernel_bench_entrypoint(
    workspace: str,
    bench_command: str,
    files_b64_json: str,
    result_glob: str,
    timeout_sec: int,
) -> str:
    """Compose the Ray Dashboard entrypoint that runs a kernel micro-
    benchmark on a GPU-bearing pod via kernel_bench_multinode.py
    (heredoc-embedded).

    Unlike apply/revert which fan out to every node, the bench runs on
    a SINGLE GPU-bearing node (the head). Multi-rank micro-benchmark
    isn't a sandbox-side concern — the per-kernel optimization loop
    scales by parallel candidate count at a higher layer.
    """
    py = _read_pod_script("kernel_bench_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kernel_bench_multinode.py\" "
        f"<<'__MN_KBENCH_PY_EOF__'\n"
        f"{py}__MN_KBENCH_PY_EOF__\n"
        f"python3 \"$WORK_DIR/kernel_bench_multinode.py\" bench "
        f"--workspace {workspace!r} "
        f"--bench-command {bench_command!r} "
        f"--files-b64-json {files_b64_json!r} "
        f"--result-glob {result_glob!r} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _extract_pod_json(logs: str) -> dict | None:
    """Parse the last top-level JSON document from a Ray Dashboard
    job_logs blob. kernel_patch_multinode.py / kernel_bench_multinode.py
    emit exactly one ``json.dumps(payload, indent=2)`` document on
    stdout; the dashboard interleaves stderr (timestamped log lines
    prefixed ``[kernel_..._multinode TS]``) so we isolate the JSON by
    finding the last ``{`` whose matching ``}`` reaches end-of-text
    after stripping trailing whitespace.
    """
    if not logs:
        return None
    text = logs.rstrip()
    end = text.rfind("}")
    if end == -1:
        return None
    # Scan backward for the matching open brace at depth 0.
    depth = 0
    in_str = False
    esc = False
    for i in range(end, -1, -1):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:end + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _submit_and_collect_pod_json(
    head_ip: str,
    entrypoint: str,
    *,
    label: str,
    poll_interval: int,
    poll_timeout: int,
) -> tuple[int, dict | None, str]:
    """Submit ``entrypoint`` to the head dashboard, poll until terminal,
    parse the per-pod JSON payload from stdout, and return
    ``(returncode, parsed_dict_or_None, full_logs)``.

    Used by cmd_apply_patch / cmd_revert_patch / cmd_kernel_bench. Keeps
    the dashboard plumbing in one place so the three commands stay
    short and parallel in shape.
    """
    with ray_dashboard.RayDashboardClient(head_ip) as ray:
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"{label} submission_id={sub_id}")

        def _fetch():
            j = ray.get_job(sub_id)
            return j, f"{label} status={j.get('status', '?')}"

        result = _short_poll(
            label=label,
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=poll_interval,
            timeout_s=poll_timeout,
        )
        status = str(result.get("status", "")).upper()
        logs = ray.get_job_logs(sub_id)
        parsed = _extract_pod_json(logs)
        if status in _TERMINAL_FAIL_STATUSES:
            return EXIT_TRANSIENT, parsed, logs
        if status not in _TERMINAL_OK_STATUSES:
            return EXIT_TRANSIENT, parsed, logs
        # SUCCEEDED on the dashboard side. Sub-script's own ``status``
        # field tells us whether the per-pod fan-out actually worked.
        if parsed is None:
            return EXIT_TRANSIENT, None, logs
        sub_status = str(parsed.get("status", "")).lower()
        return (EXIT_OK if sub_status == "ok" else EXIT_TRANSIENT), parsed, logs


# ---------------------------------------------------------------------------
# Subcommand: apply-patch / revert-patch / kernel-bench (multi-node only)


def cmd_apply_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch to every pod (head + workers).

    Read the patch file from sandbox, base64-encode it, submit a Ray
    Dashboard entrypoint that runs kernel_patch_multinode.py apply on
    the head pod; that script spawns per-node actors to write the same
    patch to ``--target-path`` on each pod.

    Multi-node only. Single-node falls back to ``apply_kernel_patch.py``
    in the sandbox (no Ray dispatch needed).

    Stdout: the same JSON document kernel_patch_multinode.py emits,
    re-printed verbatim so sandbox-side callers (apply_kernel_patch.py
    multi-node dispatch) can parse it deterministically.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_apply_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("apply-patch requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError as exc:
        err(f"failed to read patch_file {patch_path}: {exc}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_bytes).decode("ascii")

    info(
        f"apply-patch: target={args.target_path} kernel_id={args.kernel_id!r} "
        f"backup_dir={args.backup_dir} bytes={len(patch_bytes)}"
    )

    entrypoint = _build_multinode_apply_patch_entrypoint(
        args.target_path, patch_b64, args.backup_dir, args.kernel_id,
        args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        head_ip, entrypoint, label="apply-patch",
        poll_interval=args.poll_interval, poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_revert_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch revert across the pods that originally
    received it. ``--backup-map-json`` is the per-host map returned by
    the matching ``apply-patch`` call; callers MUST pass it through
    unchanged so the right backups are read on each pod.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_revert_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("revert-patch requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    try:
        decoded_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json is not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    if not decoded_map:
        err("--backup-map-json must be a non-empty {host: backup_path} object")
        return EXIT_CONFIG_ERROR

    info(
        f"revert-patch: target={args.target_path} "
        f"backup_hosts={list(decoded_map.keys())}"
    )

    entrypoint = _build_multinode_revert_patch_entrypoint(
        args.target_path, args.backup_map_json, args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        head_ip, entrypoint, label="revert-patch",
        poll_interval=args.poll_interval, poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("revert-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Fan-out the TraceLens SGLang patch set to every pod (head + workers).

    Submitted via Ray Dashboard REST as ``apply_tracelens_patch_multinode.py``;
    that script enumerates all alive nodes and spawns per-node actors that
    ``import sglang`` + ``git apply`` the per-version TraceLens patches.

    Why this command exists
    -----------------------

    Single-node Hyperloom runs ``_server_patcher.ensure_sglang_patched_for_
    tracelens`` directly because the same Python process imports SGLang
    and can locate the install root. On multi-node the controller
    (sandbox) cannot ``import sglang`` (it's only installed in the
    RayJob pods), so the local patcher silently skips and SGLang starts
    without the TraceLens annotation patches — torch.profiler then emits
    only ``step[DECODE bs=N]`` / ``step[EXTEND bs=N toks=M]`` markers
    which the TraceLens splitter does not recognise, so
    ``tracelens_analysis.py`` raises ``trace_split_no_steady_state`` and
    the orchestration agent stalls in a ``proposals=0`` loop.

    Idempotency
    -----------

    The in-pod script grep-checks the sentinel files BEFORE running
    ``git apply``. Already-patched pods return ``status=skipped`` after a
    cheap file read, so the controller can call this on every
    ``restart_server_for_round`` without worrying about double-patch.

    Stdout
    ------

    JSON document with ``status`` (``applied`` / ``skipped`` / ``failed``)
    and ``per_pod`` (list of per-host summaries). Same shape as
    ``apply_tracelens_patch_multinode.py`` emits, re-printed verbatim.

    Multi-node only.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_apply_tracelens_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err(
            "apply-tracelens-patch requires head_pod_ip in state file; "
            "run create-rayjob first"
        )
        return EXIT_CONFIG_ERROR

    tracelens_root = (
        args.tracelens_root
        or os.environ.get("TRACELENS_ROOT", "").strip()
    )
    if not tracelens_root:
        err(
            "apply-tracelens-patch requires --tracelens-root or "
            "$TRACELENS_ROOT to point at the TraceLens checkout "
            "(must be visible from every pod, typically a wekafs path)"
        )
        return EXIT_CONFIG_ERROR

    info(
        f"apply-tracelens-patch: tracelens_root={tracelens_root!r} "
        f"version_pin={args.sglang_version_pin!r}"
    )

    entrypoint = _build_multinode_apply_tracelens_patch_entrypoint(
        tracelens_root, args.sglang_version_pin or "",
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        head_ip, entrypoint, label="apply-tracelens-patch",
        poll_interval=args.poll_interval, poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-tracelens-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    # ``_submit_and_collect_pod_json`` expects the in-pod script to emit
    # ``"status": "ok"`` (kernel_patch_multinode.py convention). Our
    # tracelens patcher emits ``"applied"`` / ``"skipped"`` so the helper
    # mis-classifies a successful run as transient. Derive the exit code
    # ourselves from the parsed payload — driver_exit_code is already
    # baked into ``status``: SUCCEEDED before we reach this branch.
    if parsed.get("status") in ("applied", "skipped"):
        return EXIT_OK
    return EXIT_TRANSIENT


def cmd_kernel_bench(args: argparse.Namespace) -> int:
    """Run a kernel micro-benchmark on a GPU-bearing pod.

    Stages an optional bundle of helper files into ``--workspace``,
    invokes ``--bench-command`` under that workspace with GPU
    acceleration, and reads back result artifacts matching
    ``--result-glob``.

    The sandbox calls this when ``is_multi_node()`` is True and the
    kernel-agent micro-benchmark step would otherwise try to compile +
    run on the sandbox (which lacks GPUs in multi-node mode).
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_kernel_bench(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("kernel-bench requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    # Validate the optional files-b64 JSON early so a malformed input
    # fails before we hit the dashboard.
    if args.files_b64_json:
        try:
            json.loads(args.files_b64_json)
        except json.JSONDecodeError as exc:
            err(f"--files-b64-json is not valid JSON: {exc}")
            return EXIT_CONFIG_ERROR

    info(
        f"kernel-bench: workspace={args.workspace} "
        f"cmd={args.bench_command!r} result_glob={args.result_glob}"
    )

    entrypoint = _build_multinode_kernel_bench_entrypoint(
        args.workspace, args.bench_command, args.files_b64_json or "{}",
        args.result_glob, args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        head_ip, entrypoint, label="kernel-bench",
        poll_interval=args.poll_interval, poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("kernel-bench: could not parse pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_restart_server(args: argparse.Namespace) -> int:
    """Kill any prior vllm/sglang server and launch a new one.

    Two paths, picked from state.json's ``nodes`` field (written by
    create-rayjob):

    * ``nodes <= 1`` (single-pod) — submit a bash entrypoint that runs
      kill_server.sh + launch_server.sh on the head pod. Same as the
      pre-multinode behaviour; nothing changes for single-pod sessions.
    * ``nodes >= 2`` (multi-pod) — submit a Python entrypoint
      that uses ray actors to fan out kill_multinode.py + launch_multinode.py
      across every node, wiring sglang/vllm with --nnodes / --node-rank /
      --dist-init-addr per upstream multi-node docs. The agent runs ONE
      restart-server invocation; the driver inside the pod handles the rest.

    Dynamo backend (state.backend == 'dynamo') routes to the SSH fan-out path
    instead of the Ray Dashboard; the RayJob path below is byte-for-byte
    unchanged for non-dynamo state.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_restart_server(args)
    state = _require_state("head_pod_ip")
    head_ip = state["head_pod_ip"]
    nnodes = int(state.get("nodes") or 1)

    if nnodes >= 2:
        # Multi-node: dir-based PID/log layout (one file per rank).
        pid_dir = args.pid_file or state.get("last_server_pid_dir") or "/tmp/multi_node_pids"
        log_dir = args.log_file or state.get("last_server_log_dir") or "/tmp/multi_node_logs"
        info(f"restart-server (multi-node): framework={args.framework} "
             f"model={args.model} tp={args.tp} nnodes={nnodes}")

        kill_ep = _build_multinode_kill_entrypoint(pid_dir)
        launch_ep = _build_multinode_launch_entrypoint(args, nnodes, pid_dir, log_dir)

        # Resume-running-launch fast path: if the previous restart-server
        # invocation already submitted a launch with identical
        # framework/model/tp/ep/pd_mode AND the Ray dashboard reports
        # that job is still RUNNING, skip KILL+LAUNCH and just resume
        # polling it.
        #
        # Rationale: large MoE servers (DSr1-0528 671B FP8, TP=16) need
        # 5-10 minutes to load weights + Triton/AITER JIT compile, far
        # longer than the 110s --poll-timeout most retry loops use. The
        # legacy "kill old, launch new" path means every retry iteration
        # resets the server boot from zero — sglang/vllm can never
        # finish loading because the next 110s retry kills it again.
        #
        # Disable with MULTI_NODE_RESTART_RESUME_RUNNING=0 (operator
        # explicitly wants a fresh server even though the flags match).
        launch_sub: str = ""
        resume_enabled = (
            os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING", "1").lower()
            not in ("0", "false", "no", "off")
        )
        prev_sub = str(state.get("last_restart_submission_id") or "").strip()
        # ``last_restart_extra_args`` is normalized at write time; do the
        # same to the live args here so whitespace differences don't make
        # an otherwise-identical restart miss the resume fast path.
        # CRITICAL: extra_args carries every backend / params variant flag
        # (--attention-backend, --cuda-graph-max-bs, --moe-runner-backend,
        # etc.); a resume that ignores it leaves sglang running with the
        # PREVIOUS variant's args, so every benchmark measurement after
        # the first variant reflects stale flags instead of the round's
        # intended config.
        prev_match = bool(prev_sub) and (
            str(state.get("last_restart_framework") or "") == str(args.framework)
            and str(state.get("last_restart_model") or "") == str(args.model)
            and int(state.get("last_restart_tp") or 0) == int(args.tp)
            and int(state.get("last_restart_ep") or 1) == int(getattr(args, "ep", 1) or 1)
            and str(state.get("last_restart_pd_mode") or "colocated")
                == (getattr(args, "pd_mode", "") or "colocated").lower()
            and _normalize_extra_args(state.get("last_restart_extra_args"))
                == _normalize_extra_args(getattr(args, "extra_args", ""))
        )
        if resume_enabled and prev_match:
            _prev_status = ""
            try:
                with ray_dashboard.RayDashboardClient(head_ip) as _probe:
                    _job = _probe.get_job(prev_sub)
                    _prev_status = str(_job.get("status", "")).upper()
            except Exception as _exc:  # noqa: BLE001
                info(f"resume probe failed: {_exc!r}; falling back to KILL+LAUNCH")
            if _prev_status == "RUNNING":
                info(f"resume: reusing launch_sub={prev_sub} "
                     f"(framework={args.framework} model={args.model} "
                     f"tp={args.tp} ep={getattr(args, 'ep', 1)}) "
                     f"— skipping KILL+LAUNCH, just polling")
                launch_sub = prev_sub
            elif _prev_status in _TERMINAL_OK_STATUSES:
                info(f"resume: prior launch_sub={prev_sub} already SUCCEEDED; "
                     f"skipping KILL+LAUNCH, treating as healthy")
                launch_sub = prev_sub

        kill_sub = ""
        if not launch_sub:
            kill_sub = _exec_kill_submission(
                head_ip, kill_ep, label="restart kill", args=args,
            )

        with ray_dashboard.RayDashboardClient(head_ip) as ray:
            # Phase B: launch new (skipped above when resuming an
            # existing RUNNING launch). Driver returns once every rank
            # spawned its launcher; rank 0 /health probe is best-effort
            # (driver internal, see launch_multinode.py).
            if not launch_sub:
                launch_sub = ray.submit_job(_wrap_for_dash(launch_ep))
                info(f"launch submission_id={launch_sub} "
                     f"(driver waits for actors, then returns; servers detached)")

            # EARLY checkpoint: persist the launch identity + config NOW,
            # before the (potentially long) _short_poll. Without this,
            # the next 110s poll timeout raises TransientFailure and
            # cmd_restart_server returns early — leaving state.json
            # untouched. The retry loop's next invocation can't see
            # last_restart_submission_id, falls into KILL+LAUNCH again,
            # and the server bootstrap is reset from zero. Persisting
            # here lets the resume-running-launch fast path above
            # actually fire on retry.
            state["last_server_pid_dir"] = pid_dir
            state["last_server_log_dir"] = log_dir
            if kill_sub:
                state["last_kill_submission_id"] = kill_sub
            state["last_restart_submission_id"] = launch_sub
            state["last_restart_framework"] = args.framework
            state["last_restart_model"] = args.model
            state["last_restart_tp"] = int(args.tp)
            state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
            state["last_restart_pd_mode"] = (
                getattr(args, "pd_mode", "") or "colocated"
            ).lower()
            state["last_restart_extra_args"] = _normalize_extra_args(
                getattr(args, "extra_args", "")
            )
            _save_state(state)

            def _fetch_launch():
                j = ray.get_job(launch_sub)
                return j, f"launch status={j.get('status', '?')}"

            result = _short_poll(
                label=f"launch {launch_sub}",
                fetch=_fetch_launch,
                is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
                is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
                interval_s=args.poll_interval,
                timeout_s=_poll_timeout_from_args(args),
            )
            launch_status = str(result.get("status", "")).upper()
            if args.print_logs or launch_status in _TERMINAL_FAIL_STATUSES:
                logs = ray.get_job_logs(launch_sub)
                print(logs)

            # Fail-fast on driver terminal failure (launch_multinode.py
            # exits non-zero -> Ray job status FAILED/STOPPED). Without
            # this return, the caller (hyperloom
            # `_multi_node_server_lifecycle.restart_server_for_round`)
            # sees rc=0, skips `ServerRestartFailed`, and burns the full
            # `HYPERLOOM_MN_HEALTH_WAIT_S` (1800s) on a corpse before
            # giving up. The driver now writes
            # `MULTI_NODE_FAILURE_SNAPSHOT={...}` to stderr on rank-0
            # process early-exit; surfacing rc=1 here lets the grid
            # runner skip the broken variant in seconds.
            if launch_status in _TERMINAL_FAIL_STATUSES:
                info(f"ERROR launch driver terminal status={launch_status}; "
                     f"multi-node restart failed (sub_id={launch_sub}). "
                     f"Check stderr above for MULTI_NODE_FAILURE_SNAPSHOT.")
                return 1

            # PD disaggregated: parse the launcher's stdout JSON to get
            # internal prefill/decode URLs, then submit a *separate*
            # router entrypoint that detaches sglang_router (or vllm
            # disagg_proxy) on the head pod. Router binds the public
            # 8888 port so the magpie client URL never changes between
            # modes. Failure to start the router is fatal — without
            # it the public port has nothing serving requests.
            pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
            router_sub = ""
            router_state: dict = {}
            if pd_mode == "disaggregated":
                launch_logs = ray.get_job_logs(launch_sub)
                # launch_multinode.py writes a single JSON document
                # (multi-line indent=2) to stdout. Find its boundaries
                # by the closing `}` and parse upward.
                router_state = _extract_launcher_summary(launch_logs)
                prefill_url = str(router_state.get("pd_prefill_url") or "").strip()
                decode_url = str(router_state.get("pd_decode_url") or "").strip()
                if not prefill_url or not decode_url:
                    info("ERROR PD launcher summary missing prefill/decode URL; "
                         "cannot start router. Inspect launch logs above.")
                    return 1

                info(f"PD launcher summary: prefill={prefill_url} "
                     f"decode={decode_url}; submitting router entrypoint")
                router_ep = _build_multinode_router_entrypoint(
                    args, prefill_url, decode_url, pid_dir, log_dir,
                )
                router_sub = ray.submit_job(_wrap_for_dash(router_ep))
                info(f"router submission_id={router_sub} "
                     f"(detaches launch_router.py; dashboard exits when router is alive)")

                def _fetch_router():
                    j = ray.get_job(router_sub)
                    return j, f"router status={j.get('status', '?')}"

                router_result = _short_poll(
                    label=f"router {router_sub}",
                    fetch=_fetch_router,
                    is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
                    is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
                    interval_s=args.poll_interval,
                    timeout_s=_poll_timeout_from_args(args),
                )
                router_status = str(router_result.get("status", "")).upper()
                if args.print_logs or router_status in _TERMINAL_FAIL_STATUSES:
                    print(ray.get_job_logs(router_sub))
                if router_status in _TERMINAL_FAIL_STATUSES:
                    info(f"ERROR router job ended {router_status}; PD restart failed")
                    return 1

        state["last_server_pid_dir"] = pid_dir
        state["last_server_log_dir"] = log_dir
        state["last_kill_submission_id"] = kill_sub
        state["last_restart_submission_id"] = launch_sub
        state["last_restart_framework"] = args.framework
        state["last_restart_model"] = args.model
        state["last_restart_tp"] = args.tp
        state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
        # Persist PD state so subsequent restart-server invocations (and
        # the orchestrator helper) can fall back to the previous values
        # when the agent omitted a PD flag.
        pd_mode_persist = (getattr(args, "pd_mode", "") or "colocated").lower()
        state["last_restart_pd_mode"] = pd_mode_persist
        if pd_mode_persist == "disaggregated":
            state["last_restart_pd_prefill_nodes"] = int(
                getattr(args, "pd_prefill_nodes", 0) or 0,
            )
            state["last_restart_pd_decode_nodes"] = int(
                getattr(args, "pd_decode_nodes", 0) or 0,
            )
            state["last_restart_pd_prefill_tp"] = int(
                getattr(args, "pd_prefill_tp", 0) or 0,
            )
            state["last_restart_pd_decode_tp"] = int(
                getattr(args, "pd_decode_tp", 0) or 0,
            )
            state["last_restart_pd_transfer_backend"] = (
                getattr(args, "pd_transfer_backend", "") or ""
            )
            state["last_restart_pd_ib_device"] = (
                getattr(args, "pd_ib_device", "") or ""
            )
            state["last_router_submission_id"] = router_sub
            state["pd_prefill_url"] = router_state.get("pd_prefill_url", "")
            state["pd_decode_url"] = router_state.get("pd_decode_url", "")
        _save_state(state)
        info("multi-node servers launched; benchmarks should target $service_url")
        return 0

    # Single-node path — unchanged from pre-multinode behaviour. PD
    # disaggregation is meaningless on a single pod (no separate
    # prefill / decode hosts); fail loudly rather than silently
    # dropping the PD flags so the operator can fix the topology.
    if (getattr(args, "pd_mode", "") or "colocated").lower() == "disaggregated":
        info(
            "ERROR --pd-mode disaggregated is not supported in single-node "
            "mode (state.json says nodes=1). PD requires >=2 nodes so "
            "prefill and decode can run on disjoint pods. Re-create the "
            "RayJob with `multi_node create-rayjob --nodes >=2` before "
            "asking for PD."
        )
        return 2
    pid_file = args.pid_file or state.get("last_server_pid_file") or "/tmp/multi_node_server.pid"
    log_file = args.log_file or "/tmp/multi_node_server.log"
    entrypoint = _build_restart_entrypoint(args, pid_file, log_file)

    with ray_dashboard.RayDashboardClient(head_ip) as ray:
        info(f"restart-server (single-node): framework={args.framework} model={args.model} tp={args.tp}")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id} (entrypoint will exit after launch; server keeps running via nohup)")

        def _fetch():
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"restart {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs or str(result.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES:
            logs = ray.get_job_logs(sub_id)
            print(logs)

    state["last_server_pid_file"] = pid_file
    state["last_server_log_file"] = log_file
    state["last_restart_submission_id"] = sub_id
    state["last_restart_framework"] = args.framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = args.tp
    _save_state(state)
    info("server launch acknowledged; subsequent benchmark jobs can target $service_url")
    return 0


def cmd_kill_inference(args: argparse.Namespace) -> int:
    """Kill vllm/sglang on the RayJob without starting replacements (free GPUs)."""
    if _load_state().get("backend") == "dynamo":
        return _dynamo_kill_inference(args)
    state = _require_state("head_pod_ip")
    head_ip = state["head_pod_ip"]
    nnodes = int(state.get("nodes") or 1)

    if nnodes >= 2:
        pid_dir = args.pid_file or state.get("last_server_pid_dir") or "/tmp/multi_node_pids"
        info(f"kill-inference (multi-node): pid_dir={pid_dir}")
        kill_ep = _build_multinode_kill_entrypoint(pid_dir)
        kill_sub = _exec_kill_submission(
            head_ip, kill_ep, label="kill-inference", args=args,
        )
        state["last_kill_submission_id"] = kill_sub
        _save_state(state)
        return 0

    pid_file = args.pid_file or state.get("last_server_pid_file") or "/tmp/multi_node_server.pid"
    info(f"kill-inference (single-node): pid_file={pid_file}")
    entrypoint = _build_kill_single_entrypoint(pid_file)
    kill_sub = _exec_kill_submission(
        head_ip, entrypoint, label="kill-inference", args=args,
    )
    state["last_kill_submission_id"] = kill_sub
    _save_state(state)
    return 0


def kill_inference_for_kernel_agent_best_effort() -> None:
    """Best-effort inference teardown before kernel-agent Ray GPU tasks.

    Swallows errors so a missing RayJob state file does not abort optimization.
    """
    ns = argparse.Namespace(
        pid_file=None,
        print_logs=False,
        poll_interval=_DEFAULT_POLL_INTERVAL_S,
        poll_timeout=_resolve_poll_timeout_s(),
    )
    try:
        cmd_kill_inference(ns)
    except Exception as exc:  # noqa: BLE001
        warn(f"kill-inference skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Subcommand: stop-rayjob


def cmd_stop_rayjob(args: argparse.Namespace) -> int:
    """Stop the RayJob via SaFE REST. Idempotent."""
    state = _load_state()
    wid = args.workload_id or state.get("rayjob_id")
    if not wid:
        warn("no rayjob_id in state file and --workload-id not provided; nothing to stop")
        return 0
    with safe_client.from_env() as safe:
        info(f"stopping workload {wid} (delete={args.delete})")
        if args.delete:
            safe.delete_workload(wid)
        else:
            safe.stop_workload(wid)
    if args.clear_state:
        try:
            STATE_FILE.unlink(missing_ok=True)
            info(f"cleared {STATE_FILE}")
        except OSError as exc:
            warn(f"could not unlink {STATE_FILE}: {exc}")
    return 0


# ---------------------------------------------------------------------------
# argparse


def _add_common_poll_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--poll-interval", type=int, default=_DEFAULT_POLL_INTERVAL_S,
                   help=f"seconds between polls (default {_DEFAULT_POLL_INTERVAL_S})")
    p.add_argument(
        "--poll-timeout",
        type=int,
        default=None,
        help=(
            "max seconds before this CLI invocation gives up "
            f"(default: HYPERLOOM_MN_POLL_TIMEOUT_S env or {_DEFAULT_POLL_TIMEOUT_S}; "
            f"use 1800 for MoE JIT cold-start). Re-run the same subcommand to "
            "keep polling."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m inference_optimizer.multi_node",
        description=(
            "Manage a session-scoped SaFE RayJob for multi-node inference "
            "optimization. State persists in /tmp/multi_node_state.json."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # create-rayjob
    sp = sub.add_parser("create-rayjob", help="create the RayJob via SaFE REST")
    sp.add_argument(
        "--workspace", default=None,
        help="SaFE workspace id. Resolution: --workspace > $SAFE_WORKSPACE "
             "env. Bails fast if neither is set.",
    )
    sp.add_argument("--image", required=True, help="container image for both head and worker pods")
    sp.add_argument("--nodes", type=int, required=True, help="total node count (>=1)")
    sp.add_argument(
        "--gpus-per-node", type=int, default=8,
        help="GPUs per pod (default 8 — full MI300X / MI355X node). "
             "Override only if the user prompt explicitly asks for a "
             "smaller per-pod GPU count.",
    )
    sp.add_argument("--cpus-per-node", type=int, default=96,
                    help="default 96 — matches a full MI300X / MI355X pod. "
                         "Override only if the user prompt asks for less.")
    sp.add_argument("--mem-per-node", type=int, default=1024,
                    help="GiB per pod. default 1024 — matches a full MI300X / "
                         "MI355X pod. Override only if the user prompt asks for less.")
    sp.add_argument("--ephemeral-per-node", type=int, default=400,
                    help="GiB per pod. default 400.")
    sp.add_argument(
        "--display-name", default=None,
        help="Optional human-readable RayJob name (shows up in SaFE UI). "
             "Resolution: $DISPLAY_NAME env > --display-name > "
             "auto-generated multi_node_<unix-ts>.",
    )
    sp.add_argument("--description", default=None)
    sp.add_argument(
        "--owner-id", default=None,
        help="ownerId for SaFE cascading cleanup. Resolution: --owner-id > "
             "$WORKLOAD_ID (sandbox workload id). When set, SaFE GCs the "
             "RayJob when the owner workload stops (safety net for missed "
             "`stop-rayjob`).",
    )
    sp.add_argument("--extra-env", action="append", default=[],
                    help="K=V (repeatable); merged AFTER credential fanout")
    sp.add_argument("--extra-label", action="append", default=[],
                    help="K=V (repeatable); reserved primus-safe.* prefixes are stripped")
    sp.add_argument("--no-wait", action="store_true",
                    help="don't poll for Running; just create and exit")
    sp.add_argument("--recreate", action="store_true",
                    help="force creating a fresh workload even if state.json "
                         "already has a live rayjob_id. Default behaviour is "
                         "to REUSE the prior workload (idempotent retries).")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_create_rayjob)

    # create-dynamo (Dynamo idle-pod backend)
    sp = sub.add_parser(
        "create-dynamo",
        help="create an idle multi-node DynamoDeployment (SSH control plane); "
             "benchmark entry point is the Dynamo frontend :8000",
    )
    sp.add_argument("--workspace", default=None,
                    help="SaFE workspace id (--workspace > $SAFE_WORKSPACE)")
    sp.add_argument("--image", required=True,
                    help="dynamo image WITH the sshd layer (mn-idle.sh present)")
    sp.add_argument("--nodes", type=int, required=True,
                    help="worker LWS node count (>=1); worker.replica == nodes")
    sp.add_argument("--gpus-per-node", type=int, default=8)
    sp.add_argument("--cpus-per-node", type=int, default=96)
    sp.add_argument("--mem-per-node", type=int, default=1024, help="GiB per worker pod")
    sp.add_argument("--ephemeral-per-node", type=int, default=400, help="GiB per worker pod")
    sp.add_argument("--shared-mem-per-node", type=int, default=200,
                    help="GiB /dev/shm per worker pod (sharedMemory)")
    sp.add_argument("--backend-framework", default="sglang",
                    choices=("sglang", "vllm", "trtllm"))
    sp.add_argument("--kv-transfer-backend", default="nixl",
                    choices=("nixl", "mori", "mooncake"))
    sp.add_argument("--ssh-port", type=int, default=ssh_client.DEFAULT_SSH_PORT,
                    help=f"pod sshd port (default {ssh_client.DEFAULT_SSH_PORT}; "
                         f"not 22 to avoid hostNetwork collision)")
    sp.add_argument("--pd-mode", choices=("aggregated", "disaggregated"),
                    default="aggregated",
                    help="aggregated [frontend,worker] (default) or "
                         "disaggregated [frontend,prefill,decode]")
    sp.add_argument("--pd-prefill-nodes", type=int, default=0,
                    help="prefill role replica (disaggregated only)")
    sp.add_argument("--pd-decode-nodes", type=int, default=0,
                    help="decode role replica (disaggregated only)")
    sp.add_argument("--pd-prefill-tp", type=int, default=0,
                    help="prefill TP; a role spans nodes (LWS) when tp > gpus-per-node")
    sp.add_argument("--pd-decode-tp", type=int, default=0,
                    help="decode TP; a role spans nodes (LWS) when tp > gpus-per-node")
    sp.add_argument("--display-name", default=None)
    sp.add_argument("--description", default=None)
    sp.add_argument("--owner-id", default=None)
    sp.add_argument("--extra-env", action="append", default=[],
                    help="K=V (repeatable); merged AFTER credential fanout")
    sp.add_argument("--extra-label", action="append", default=[])
    sp.add_argument("--no-wait", action="store_true")
    sp.add_argument("--recreate", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_create_dynamo)

    # bootstrap
    sp = sub.add_parser("bootstrap",
        help="install OOB/CLI toolchain inside the RayJob via Ray Dashboard REST. "
             "Default: uses multi_node/scripts/bootstrap.sh from this checkout.")
    sp.add_argument("--script", default=None,
                    help="optional override: absolute path to a bootstrap.sh "
                         "ALREADY VISIBLE inside the RayJob pod (e.g. baked into "
                         "the image). When omitted, the bundled script is streamed in.")
    sp.add_argument("--force", action="store_true",
                    help="re-run bootstrap even if the marker file says it's done")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_bootstrap)

    # verify
    sp = sub.add_parser("verify", help="confirm oob/claude/codex/ray are on PATH inside the RayJob")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_verify)

    # restart-server
    sp = sub.add_parser("restart-server", help="kill prior vllm/sglang server and start a new one")
    sp.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    sp.add_argument("--model", required=True, help="model path or HF id")
    sp.add_argument("--tp", type=int, required=True)
    sp.add_argument(
        "--ep", type=int, default=1,
        help="expert-parallel size for MoE inference. 1 (default) keeps "
             "experts sharded by TP. >=2 enables EP: sglang gets "
             "`--enable-ep-moe --ep-size N`, vllm gets "
             "`--enable-expert-parallel`. EP > TP is rejected by the "
             "orchestrator helper before this CLI is invoked.",
    )
    # Prefill-Decode disaggregation. colocated (default) keeps the
    # legacy single-server-group behaviour. disaggregated splits the
    # cluster into prefill + decode groups, launches sglang_router /
    # vllm proxy on the head pod, and binds the public 8888 port at
    # the router (so magpie's BENCHMARK_BASE_URL stays unchanged).
    sp.add_argument(
        "--pd-mode", choices=("colocated", "disaggregated"),
        default="colocated",
        help="PD disaggregation mode (default colocated).",
    )
    sp.add_argument(
        "--pd-prefill-nodes", type=int, default=0,
        help="number of prefill nodes (disaggregated only); pn+dn==nnodes",
    )
    sp.add_argument(
        "--pd-decode-nodes", type=int, default=0,
        help="number of decode nodes (disaggregated only)",
    )
    sp.add_argument(
        "--pd-prefill-tp", type=int, default=0,
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    sp.add_argument(
        "--pd-decode-tp", type=int, default=0,
        help="TP for decode group (disaggregated only); default = --tp",
    )
    # Per-role EP / extra server args (disaggregated only). The InferenceX
    # disagg recipes give prefill and decode DIFFERENT MoE topologies
    # (e.g. prefill EP1 no-DP vs decode EP8 DP-attn), so a single shared
    # --ep / --extra-args cannot express them. These per-role knobs default
    # to 0 / "" (fall back to the shared --ep / --extra-args, preserving the
    # legacy behaviour) and read $PD_*_EP / $PD_*_EXTRA_ARGS env as defaults.
    sp.add_argument(
        "--pd-prefill-ep", type=int,
        default=int(os.environ.get("PD_PREFILL_EP", "0") or 0),
        help="EP for prefill group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-decode-ep", type=int,
        default=int(os.environ.get("PD_DECODE_EP", "0") or 0),
        help="EP for decode group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-prefill-extra-args",
        default=os.environ.get("PD_PREFILL_EXTRA_ARGS", ""),
        help="extra sglang args appended to the PREFILL group only "
             "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-decode-extra-args",
        default=os.environ.get("PD_DECODE_EXTRA_ARGS", ""),
        help="extra sglang args appended to the DECODE group only "
             "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-transfer-backend", default="",
        help="sglang: mooncake|nixl ; vllm: NixlConnector|P2pNcclConnector|"
             "MooncakeConnector|LMCacheConnectorV1; empty = framework default",
    )
    sp.add_argument(
        "--pd-ib-device", default="",
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). "
             "Empty = read $NCCL_IB_HCA from RayJob pod env.",
    )
    sp.add_argument(
        "--pd-bootstrap-port", type=int, default=8998,
        help="sglang PD bootstrap rendezvous port (default 8998)",
    )
    sp.add_argument(
        "--pd-vllm-router-cmd", default="",
        help="vllm-only override for router cmdline; supports {prefill}/{decode}/{port} placeholders",
    )
    sp.add_argument("--extra-args", default="", help="extra CLI args appended verbatim to the framework launch command")
    sp.add_argument("--pid-file", default=None,
                    help="PID file path inside the head pod; defaults to /tmp/multi_node_server.pid")
    sp.add_argument("--log-file", default=None,
                    help="server log path inside the head pod; defaults to /tmp/multi_node_server.log")
    sp.add_argument("--no-wait-health", action="store_true",
                    help="exit the dashboard job immediately after launch instead of waiting for /health")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_restart_server)

    sp = sub.add_parser(
        "kill-inference",
        help="kill vllm/sglang on the RayJob only (no new server; frees GPU for kernel-agent)",
    )
    sp.add_argument(
        "--pid-file",
        default=None,
        help="single-node: head-pod PID file path; multi-node: pid-dir path "
             "(same override semantics as restart-server --pid-file)",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_kill_inference)

    # stop-rayjob
    sp = sub.add_parser("stop-rayjob", help="stop the RayJob via SaFE REST")
    sp.add_argument("--workload-id", default=None, help="override the workload id from state file")
    sp.add_argument("--delete", action="store_true",
                    help="hard delete instead of soft stop (default: stop)")
    sp.add_argument("--clear-state", action="store_true",
                    help="remove /tmp/multi_node_state.json on success")
    sp.set_defaults(func=cmd_stop_rayjob)

    # apply-patch (multi-node only)
    sp = sub.add_parser(
        "apply-patch",
        help="fan-out a kernel patch to every pod (head + workers); multi-node only",
    )
    sp.add_argument("--patch-file", required=True,
                    help="path to the patch source on sandbox filesystem; contents will be base64-encoded into the dashboard entrypoint")
    sp.add_argument("--target-path", required=True,
                    help="absolute file path on each pod to overwrite (e.g. /sgl-workspace/aiter/aiter/ops/gemm.py)")
    sp.add_argument("--backup-dir", required=True,
                    help="directory on each pod where the pre-patch original is saved (e.g. /var/kernel_patch_backups)")
    sp.add_argument("--kernel-id", default="",
                    help="optional id used to construct backup filename")
    sp.add_argument("--timeout-sec", type=int, default=120,
                    help="per-actor timeout (default 120s)")
    sp.add_argument("--print-logs", action="store_true",
                    help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_patch)

    # revert-patch (multi-node only)
    sp = sub.add_parser(
        "revert-patch",
        help="fan-out a kernel patch revert; multi-node only",
    )
    sp.add_argument("--target-path", required=True)
    sp.add_argument("--backup-map-json", required=True,
                    help='JSON object {hostname: backup_path} from the matching apply-patch result')
    sp.add_argument("--timeout-sec", type=int, default=60)
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_revert_patch)

    # apply-tracelens-patch (multi-node only)
    sp = sub.add_parser(
        "apply-tracelens-patch",
        help=(
            "fan-out the TraceLens SGLang patch set to every pod; "
            "multi-node only. Idempotent: skips pods already patched."
        ),
    )
    sp.add_argument(
        "--tracelens-root", default=None,
        help=(
            "absolute path to public TraceLens checkout (must be visible "
            "from every pod, typically /wekafs/...). Defaults to "
            "$TRACELENS_ROOT."
        ),
    )
    sp.add_argument(
        "--sglang-version-pin", default=None,
        help=(
            "advisory pin (e.g. '0.5.11'); logged on mismatch with the "
            "sglang installed in the pod. Optional."
        ),
    )
    sp.add_argument("--print-logs", action="store_true",
                    help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_tracelens_patch)

    # kernel-bench (multi-node only)
    sp = sub.add_parser(
        "kernel-bench",
        help="run a kernel micro-benchmark on a GPU-bearing pod; multi-node only",
    )
    sp.add_argument("--workspace", required=True,
                    help="absolute dir on pod that will be CWD for the bench")
    sp.add_argument("--bench-command", required=True,
                    help="shell command to invoke (passed to 'bash -lc')")
    sp.add_argument("--files-b64-json", default="{}",
                    help='JSON {rel_path: base64_content} of helper files to stage into workspace before the bench')
    sp.add_argument("--result-glob", default="*.json",
                    help="glob (relative to workspace) of result artifacts to read back after the bench")
    sp.add_argument("--timeout-sec", type=int, default=600,
                    help="hard timeout for the bench command (default 600s)")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_kernel_bench)

    # install-geak (dynamo only)
    sp = sub.add_parser(
        "install-geak",
        help="install the GEAK CLI on every Dynamo GPU pod over SSH "
             "(idempotent; pip-installs the shared-FS GEAK checkout); dynamo only",
    )
    sp.add_argument("--geak-src", default=None,
                    help="GEAK source dir on the shared mount (default: "
                         "$HYPERLOOM_GEAK_SRC > $HYPERLOOM_ROOT/geak > "
                         "$USER_DATA_PATH/runtime/geak)")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_install_geak)

    # install-oob (dynamo only)
    sp = sub.add_parser(
        "install-oob",
        help="install the OOB backend (oob/claude/codex/@cursor) on every "
             "Dynamo GPU pod over SSH (idempotent); dynamo only",
    )
    sp.add_argument("--oob-src", default=None,
                    help="OOB source dir on the shared mount (default: $OOB_SRC)")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_install_oob)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint with stable exit codes for the hyperloom controller:

    * 0   — success
    * 1   — transient/unknown failure (poll timeout, network error,
            unexpected exception). Caller MAY rerun the same subcommand.
    * 2   — workload entered Failed/Stopped/Cancelled. DO NOT rerun;
            the cluster is unusable. The structured failure snapshot is
            logged on stderr (and emitted as a single-line JSON marker
            ``MULTI_NODE_FAILURE_SNAPSHOT={...}`` for easy parsing).
    * 3   — config error: missing env / required arg / bad input. Rerun
            won't help; fix the inputs.
    * 130 — Ctrl-C / SIGINT
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        err("interrupted")
        return EXIT_INTERRUPT
    except WorkloadTerminalFailure as wtf:
        err(str(wtf))
        # Emit a parseable single-line JSON marker so a hyperloom controller
        # tailing stderr can grep for MULTI_NODE_FAILURE_SNAPSHOT= and
        # parse the rest of the line as JSON without prying open the
        # stream.
        err("MULTI_NODE_FAILURE_SNAPSHOT=" + json.dumps(wtf.snapshot, default=str))
        return EXIT_TERMINAL_FAILURE
    except TransientFailure as tf:
        err(str(tf))
        return EXIT_TRANSIENT
    except safe_client.SafeApiError as sae:
        # SaFE returned a non-2xx HTTP status. 4xx = caller's body is
        # broken (image not found, quota exceeded, workspace missing,
        # invalid label/env) — retrying with the same args will fail
        # the same way; tell the controller not to retry (exit 3).
        # 5xx = SaFE-side glitch; safe to retry (exit 1).
        # Authentication failures (401/403) are also caller-fixable
        # (bad / missing SAFE_API_KEY) — config error.
        err(str(sae))
        if sae.status is not None and 400 <= sae.status < 500:
            return EXIT_CONFIG_ERROR
        return EXIT_TRANSIENT
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: from _require_state / from_env / unhandled SaFE error
        # ValueError: from argparse-derived input validation
        # Both are caller-fixable; classify as config error.
        msg = str(exc)
        if any(s in msg for s in (
            "is required",
            "Missing required environment variable",
            "missing required keys",
            "unsupported framework",
        )):
            err(f"{type(exc).__name__}: {exc}")
            return EXIT_CONFIG_ERROR
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT
    except Exception as exc:  # noqa: BLE001
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT


if __name__ == "__main__":
    sys.exit(main())
