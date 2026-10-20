# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node-only: proving a restarted inference server serves traffic, and the failure that says it never will."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

from hyperloom.common.timeutil import format_exc_brief
from hyperloom.common.env import env_flag, env_int
from hyperloom.inference_optimizer.multi_node._internal.serving_probe import generated_tokens
from ._multi_node_env import _read_state
from ._multi_node_worker_logs import (
    shared_server_log_dir,
    worker_detokenizer_wedged,
    worker_startup_crashed,
)

log = logging.getLogger(__name__)


class ServerRestartFailed(RuntimeError):
    """Raised when the per-round multi-node server restart did not succeed."""


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a poll-count knob whose smallest meaningful value is *minimum*, not zero."""
    return max(minimum, env_int(name, default))


class _WorkerTarget(NamedTuple):
    """One pod whose own /health gates the frontend serving probe."""

    role: str
    ip: str
    pod_id: str


def _worker_targets(state: Mapping[str, Any]) -> list[_WorkerTarget]:
    """Every distinct prefill/decode/worker pod state.json records, in role order."""
    seen: set[str] = set()
    targets: list[_WorkerTarget] = []
    for role_key in ("prefill_pods", "decode_pods", "worker_pods"):
        for pod in state.get(role_key) or []:
            if not isinstance(pod, dict):
                continue
            ip = str(pod.get("podIP") or "").strip()
            if not ip or ip in seen:
                continue
            seen.add(ip)
            targets.append(_WorkerTarget(role_key.replace("_pods", ""), ip, str(pod.get("podId") or ip)))
    return targets


@dataclass(frozen=True)
class _WorkerGateRules:
    """What the worker gate dials, and when a pending worker's own log may end the wait instead of its /health."""

    health_port: int
    fast_fail: bool
    crash_grace_s: int
    wedge_grace_s: int


def _worker_gate_rules() -> _WorkerGateRules:
    """Resolve the worker gate's port and fast-fail windows from the environment.

    Read once before the gate opens its client, so a malformed value fails the round before any worker is dialled
    rather than mid-poll or -- when the fast-fail is switched off -- not at all.
    """
    return _WorkerGateRules(
        health_port=env_int("HYPERLOOM_MN_WORKER_PORT", 30000),
        fast_fail=env_flag("HYPERLOOM_MN_WORKER_WEDGE_FASTFAIL", default=True),
        # An argparse rejection or a fatal exit is terminal, so bail within seconds of the launch.
        crash_grace_s=env_int("HYPERLOOM_MN_WORKER_CRASH_GRACE_S", 45),
        wedge_grace_s=env_int("HYPERLOOM_MN_WORKER_WEDGE_GRACE_S", 420),
    )


async def _workers_passing_health(
    client: Any,
    targets: list[_WorkerTarget],
    ready: set[str],
    elapsed: int,
    port: int,
) -> set[str]:
    """The worker IPs passing their own /health after one more sweep over the ones still pending."""
    ready = set(ready)
    for target in targets:
        if target.ip in ready:
            continue
        try:
            resp = await client.get("http://" + target.ip + ":" + str(port) + "/health")
        except Exception:  # noqa: BLE001
            continue
        if resp.status_code != 200:
            continue
        ready.add(target.ip)
        log.info(
            "worker /health OK (%s %s) after %ds [%d/%d]", target.role, target.ip, elapsed, len(ready), len(targets)
        )
    return ready


def _reject_crashed_worker(shared_dir: str, pending_ips: list[str], *, elapsed: int, timeout_s: int) -> None:
    """Fail the wait on the first pending worker whose log shows a non-recoverable startup failure."""
    for ip in pending_ips:
        reason = worker_startup_crashed(shared_dir, ip)
        if reason:
            raise ServerRestartFailed(
                "worker " + ip + " server crashed on startup "
                "(non-recoverable; /health will never come up): "
                + reason
                + " -- fast-failed after "
                + str(elapsed)
                + "s instead of the full "
                + str(timeout_s)
                + "s gate"
            )


def _reject_wedged_workers(shared_dir: str, pending_ips: list[str], *, elapsed: int, timeout_s: int) -> None:
    """Fail the wait when pending workers loaded their weights but their detokenizer never heartbeats."""
    wedged = [ip for ip in pending_ips if worker_detokenizer_wedged(shared_dir, ip)]
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


def _reject_terminal_worker_state(
    pending_ips: list[str],
    rules: _WorkerGateRules,
    *,
    elapsed: int,
    timeout_s: int,
) -> None:
    """Fail the wait early on a worker state /health can never recover from, rather than burning the whole gate."""
    if not rules.fast_fail:
        return
    shared_dir = shared_server_log_dir()
    if not shared_dir:
        return
    if elapsed > rules.crash_grace_s:
        _reject_crashed_worker(shared_dir, pending_ips, elapsed=elapsed, timeout_s=timeout_s)
    if elapsed > rules.wedge_grace_s:
        _reject_wedged_workers(shared_dir, pending_ips, elapsed=elapsed, timeout_s=timeout_s)


async def _wait_for_workers_ready_async(timeout_s: int, poll_every_s: int = 10) -> None:
    """Wait for every prefill/decode worker's own /health to return 200 before the frontend serving probe runs."""
    import time as _t

    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover
        return
    targets = _worker_targets(_read_state() or {})
    if not targets:
        return
    rules = _worker_gate_rules()
    started = _t.monotonic()
    ready: set[str] = set()
    log.info("waiting for %d worker(s) /health on port %d before frontend probe", len(targets), rules.health_port)
    async with _httpx.AsyncClient(timeout=10.0) as client:
        while True:
            elapsed = int(_t.monotonic() - started)
            ready = await _workers_passing_health(client, targets, ready, elapsed, rules.health_port)
            pending = [target.ip for target in targets if target.ip not in ready]
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
            _reject_terminal_worker_state(pending, rules, elapsed=elapsed, timeout_s=timeout_s)
            await asyncio.sleep(poll_every_s)


@dataclass(frozen=True)
class _ServingUrls:
    """The endpoints the post-restart readiness wait dials on one server."""

    health: str
    models: str
    completions: str


def _serving_urls(service_url: str) -> _ServingUrls:
    """Resolve the /health, /v1/models and /v1/completions endpoints of ``service_url``."""
    base = service_url.rstrip("/")
    return _ServingUrls(base + "/health", base + "/v1/models", base + "/v1/completions")


@dataclass(frozen=True)
class _ServingProbeRules:
    """What the post-restart wait has to see before it calls a server ready."""

    #: Infera registers models in /v1/models before its workers accept requests, so there /health alone is not
    #: readiness: the first benchmark would get 503 "Model temporarily unavailable" for every request, surfacing
    #: as ``completed=0`` → ``baseline_failed``.
    wait_model_ready: bool

    #: All three are counts of something the server has to actually produce, so none drops below 1: at zero each in
    #: turn inverts the gate, calling a server ready on a probe that failed, that was never sent, or that came back
    #: empty.
    required_streak: int
    max_tokens: int
    min_tokens: int


def _serving_probe_rules(backend: str) -> _ServingProbeRules:
    """Resolve what ``backend`` has to prove, and how often, from the environment."""
    return _ServingProbeRules(
        wait_model_ready=backend == "infera",
        required_streak=_env_int("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", 2),
        # Probe with >1 token + ignore_eos so the request must traverse the decode leg (prefill alone can serve the
        # first token): a PD run with an unready or KV-broken decode then fails the probe instead of passing on a
        # prefill-only 200.
        max_tokens=_env_int("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", 8),
        min_tokens=_env_int("HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS", 2),
    )


@dataclass(frozen=True)
class _ServingGrace:
    """How long each stage behind /health may stall before the wait fast-fails."""

    models_empty_s: int
    completions_s: int


def _serving_grace() -> _ServingGrace:
    """Resolve the post-/health fast-fail windows from the environment."""
    return _ServingGrace(
        # /health up but /v1/models never registering means the GPU workers crashed on launch (e.g. an unrecognized
        # server flag) rather than slowly loading.
        models_empty_s=int(os.environ.get("HYPERLOOM_MN_MODELS_EMPTY_GRACE_S", "600") or 600),
        # /v1/models populated but /v1/completions never serving means the PD prefill<->decode KV handoff broke on
        # the restart -> HTTP 503 on every probe.
        completions_s=int(os.environ.get("HYPERLOOM_MN_COMPLETIONS_GRACE_S", "480") or 480),
    )


@dataclass
class _ServingReadiness:
    """How far the restarted server has come, and what its last probe reported."""

    health_ok_at: int | None = None
    models_ready_at: int | None = None
    consecutive_completion_ok: int = 0
    last_err: str = ""


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


async def _run_serving_probe(
    client: Any,
    url: str,
    model_id: str,
    rules: _ServingProbeRules,
    state: _ServingReadiness,
) -> int:
    """Run one tiny completion, move the consecutive-success streak, and return how many tokens it generated."""
    try:
        cresp = await client.post(
            url,
            json={
                "model": model_id,
                "prompt": "hi",
                "max_tokens": rules.max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            },
        )
    except Exception as cexc:  # noqa: BLE001
        state.consecutive_completion_ok = 0
        state.last_err = f"completion_probe {format_exc_brief(cexc, limit=80)}"
        return 0
    if cresp.status_code != 200:
        state.consecutive_completion_ok = 0
        state.last_err = f"completion_probe_http={cresp.status_code}"
        return 0
    try:
        gen_toks = generated_tokens(cresp.json())
    except ValueError:
        gen_toks = 0
    if gen_toks < rules.min_tokens:
        # HTTP 200 but the decode leg produced nothing (unready / broken PD KV handoff) — do NOT count as ready.
        state.consecutive_completion_ok = 0
        state.last_err = (
            f"completion_probe_zero_tokens gen={gen_toks} (need>={rules.min_tokens}; decode leg not serving)"
        )
        return gen_toks
    state.consecutive_completion_ok += 1
    state.last_err = (
        f"completion_probe ok x{state.consecutive_completion_ok}/{rules.required_streak} (gen_tokens={gen_toks})"
    )
    return gen_toks


async def _models_serve_traffic(
    client: Any,
    urls: _ServingUrls,
    rules: _ServingProbeRules,
    state: _ServingReadiness,
    elapsed: int,
) -> bool:
    """Whether /v1/models lists a worker the serving probe can now reach often enough to call the server ready."""
    try:
        mresp = await client.get(urls.models)
    except Exception as mexc:  # noqa: BLE001
        state.last_err = f"models_probe {format_exc_brief(mexc, limit=80)}"
        return False
    if mresp.status_code != 200:
        state.last_err = f"models_http_status={mresp.status_code}"
        return False
    try:
        data = mresp.json()
    except ValueError:
        data = {}
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list) or not models:
        state.last_err = f"models_empty (health_ok_at={state.health_ok_at}s)"
        state.consecutive_completion_ok = 0
        return False
    if state.models_ready_at is None:
        state.models_ready_at = elapsed
        log.info(
            "post-restart /v1/models populated after %ds (n=%d, health_ok_at=%ds); now probing /v1/completions",
            elapsed,
            len(models),
            state.health_ok_at,
        )
    model_id = str(models[0].get("id") or "") if isinstance(models[0], dict) else ""
    if not model_id:
        state.last_err = "completion_probe: no model id"
        state.consecutive_completion_ok = 0
        return False
    gen_toks = await _run_serving_probe(client, urls.completions, model_id, rules, state)
    if state.consecutive_completion_ok < rules.required_streak:
        return False
    log.info(
        "post-restart READY after %ds (n=%d, models_at=%ds, completion_ok_x%d, gen_tokens=%d)",
        elapsed,
        len(models),
        state.models_ready_at,
        state.consecutive_completion_ok,
        gen_toks,
    )
    return True


async def _probe_server_ready(
    client: Any,
    urls: _ServingUrls,
    rules: _ServingProbeRules,
    state: _ServingReadiness,
    elapsed: int,
) -> bool:
    """Whether the restarted server now meets every readiness rule its backend asks for."""
    try:
        resp = await client.get(urls.health)
    except Exception as exc:  # noqa: BLE001
        state.last_err = format_exc_brief(exc, limit=120)
        return False
    if resp.status_code != 200:
        state.last_err = f"http_status={resp.status_code}"
        return False
    if state.health_ok_at is None:
        state.health_ok_at = elapsed
        log.info(
            "post-restart /health OK after %ds (url=%s)%s",
            elapsed,
            urls.health,
            "; now also waiting for /v1/models to be non-empty" if rules.wait_model_ready else "",
        )
    if not rules.wait_model_ready:
        return True
    return await _models_serve_traffic(client, urls, rules, state, elapsed)


def _reject_stalled_serving(
    urls: _ServingUrls,
    grace: _ServingGrace,
    state: _ServingReadiness,
    elapsed: int,
) -> None:
    """Fail the wait when /health is up but the stage behind it stopped making progress."""
    if _models_empty_too_long(
        elapsed=elapsed,
        health_ok_at=state.health_ok_at,
        models_ready_at=state.models_ready_at,
        grace_s=grace.models_empty_s,
    ):
        raise ServerRestartFailed(
            f"/v1/models still empty {elapsed - (state.health_ok_at or 0)}s after /health "
            f"(> grace {grace.models_empty_s}s); GPU workers crashed on launch "
            f"(url={urls.models}, last_err={state.last_err})"
        )
    if (
        grace.completions_s > 0
        and state.models_ready_at is not None
        and (elapsed - state.models_ready_at) > grace.completions_s
    ):
        raise ServerRestartFailed(
            f"/v1/models populated at {state.models_ready_at}s but /v1/completions still "
            f"not serving {elapsed - state.models_ready_at}s later (> grace {grace.completions_s}s); "
            f"serving broken, likely PD KV handoff on restart "
            f"(url={urls.completions}, last_err={state.last_err})"
        )


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
    if not service_url:
        log.warning("no service_url in state; skipping post-restart /health wait")
        return

    urls = _serving_urls(service_url)
    rules = _serving_probe_rules(str(state.get("backend") or "").strip().lower())
    grace = _serving_grace()
    progress = _ServingReadiness()
    started = _time.monotonic()
    log.info(
        "post-restart /health wait: url=%s timeout_s=%d poll_every_s=%d",
        urls.health,
        timeout_s,
        poll_every_s,
    )
    async with _httpx.AsyncClient(timeout=15.0) as client:
        while True:
            elapsed = int(_time.monotonic() - started)
            if await _probe_server_ready(client, urls, rules, progress, elapsed):
                return
            _reject_stalled_serving(urls, grace, progress, elapsed)
            if elapsed > timeout_s:
                raise ServerRestartFailed(
                    f"server /health did not return 200 within {timeout_s}s "
                    f"(url={urls.health}, last_err={progress.last_err})"
                )
            if elapsed % 60 < poll_every_s:
                log.info(
                    "post-restart /health still waiting t=%ds last_err=%s",
                    elapsed,
                    progress.last_err,
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
    return env_int("HYPERLOOM_MN_PUBLISHED_READY_S", 300)


def _rewritten_published_url(state: Mapping[str, Any]) -> str:
    """The published ``service_url`` when it differs from the reachable one, else ``""``.

    Without a rewrite there is nothing new to prove: the reachable wait already covered that exact endpoint.
    """
    from hyperloom.inference_optimizer.multi_node._internal.external_state import reachable_service_url

    if str(state.get("backend") or "").strip().lower() != "rayjob":
        return ""
    published = str(state.get("service_url") or "").strip().rstrip("/")
    return "" if not published or published == reachable_service_url(state).rstrip("/") else published


def _log_published_gate_skip(health_url: str, dns_failures: int, poll_every_s: int) -> None:
    """Record that an unresolvable published name, not a passing probe, is what ended the gate."""
    # Surface, don't degrade silently: to the caller this is indistinguishable from a pass, so a later completed=0 /
    # magpie_nonzero_invalid_measurement has no other clue it was this skip.
    log.warning(
        "published service_url %s did not resolve from here after %d consecutive "
        "attempts (~%ds); SKIPPING the published-service readiness gate. If this is "
        "an in-cluster run this is a DNS problem, and a following benchmark "
        "completed=0 / ECONNREFUSED likely traces back to this skip.",
        health_url,
        dns_failures,
        dns_failures * poll_every_s,
    )


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

    published = _rewritten_published_url(_read_state() or {})
    if not published:
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
                        _log_published_gate_skip(health_url, dns_failures, poll_every_s)
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


async def wait_for_restarted_server(health_wait_s: int) -> None:
    """Hold until the workers, the reachable frontend and the published service all report ready.

    ``cmd_restart_server`` returns once the actors are spawned, but a cold MoE weight-load can need 20-30 min
    before /health flips, so the downstream baseline must not fire against a not-yet-ready server.
    """
    poll_every_s = int(os.environ.get("HYPERLOOM_MN_HEALTH_POLL_S", "10"))
    # PD restart: ensure BOTH prefill+decode legs are /health-ready (mooncake init done) before the frontend
    # completions probe, so its grace does not expire against a half-ready pair.
    await _wait_for_workers_ready_async(timeout_s=health_wait_s, poll_every_s=poll_every_s)
    await _wait_for_server_health_async(timeout_s=health_wait_s, poll_every_s=poll_every_s)
    # The reachable /health above can be a pod-pinned head address; the benchmark dials the published ClusterIP
    # Service, whose endpoints lag readiness after a restart.
    await _wait_for_published_service_ready_async(
        timeout_s=_published_ready_timeout_s(),
        poll_every_s=poll_every_s,
    )
