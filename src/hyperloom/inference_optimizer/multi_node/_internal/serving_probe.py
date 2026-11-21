"""Single-shot checks for whether a multi-node cluster is actually serving."""

from __future__ import annotations

import logging
import time
from typing import Any

from hyperloom_common.env import env_int

from .external_state import reachable_service_url

log = logging.getLogger(__name__)


def completion_probe_tokens() -> int:
    """``max_tokens`` a ``/v1/completions`` readiness probe requests.

    Shared with the orchestrator's post-restart wait so a resume and the wait that follows it agree on what
    "serving" means.
    """
    return env_int("HYPERLOOM_MN_COMPLETION_PROBE_TOKENS", 8)


def completion_probe_min_tokens() -> int:
    """Fewest generated tokens that count a ``/v1/completions`` readiness probe as served."""
    return env_int("HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS", 2)


def generated_tokens(data: object) -> int:
    """Best-effort count of tokens a ``/v1/completions`` probe actually produced."""
    if not isinstance(data, dict):
        return 0
    usage = data.get("usage")
    if isinstance(usage, dict):
        try:
            count = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return 1 if str(choices[0].get("text") or "").strip() else 0
    return 0


def _serving_legs(state: dict[str, Any], pd_mode: str) -> list[str]:
    """The endpoints that must each be up before the cluster counts as serving."""
    if pd_mode == "disaggregated":
        legs = [
            str(state.get("pd_prefill_url") or "").strip(),
            str(state.get("pd_decode_url") or "").strip(),
        ]
        return legs if all(legs) else []
    front = reachable_service_url(state)
    return [front] if front else []


def cluster_is_serving(state: dict[str, Any], *, pd_mode: str, timeout_s: int) -> bool:
    """Whether every leg is up AND the model answers with real tokens, right now.

    Raises:
        EnvValueError: A completion-probe token setting is set but is not an integer.
    """
    wanted = completion_probe_tokens()
    floor = completion_probe_min_tokens()
    legs = _serving_legs(state, pd_mode)
    front = reachable_service_url(state)
    if not legs or not front:
        return False
    import httpx

    deadline = time.monotonic() + timeout_s

    def _left() -> float:
        """Seconds still available to the probe."""
        return deadline - time.monotonic()

    try:
        with httpx.Client() as client:
            for leg in legs:
                if _left() <= 0:
                    log.info("serving probe: %ds budget spent before %s answered", timeout_s, leg.rstrip("/"))
                    return False
                resp = client.get(leg.rstrip("/") + "/health", timeout=_left())
                if resp.status_code != 200:
                    log.info("serving probe: %s/health returned %s", leg.rstrip("/"), resp.status_code)
                    return False

            if _left() <= 0:
                log.info("serving probe: %ds budget spent before /v1/models answered", timeout_s)
                return False
            models = client.get(front.rstrip("/") + "/v1/models", timeout=_left())
            if models.status_code != 200:
                log.info("serving probe: /v1/models returned %s", models.status_code)
                return False
            body = models.json()
            entries = body.get("data") if isinstance(body, dict) else None
            if not isinstance(entries, list) or not entries:
                log.info("serving probe: /v1/models is empty; no model has registered yet")
                return False
            model_id = str(entries[0].get("id") or "") if isinstance(entries[0], dict) else ""
            if not model_id:
                log.info("serving probe: /v1/models carries no model id")
                return False

            # ignore_eos plus several tokens forces the request through the decode leg; prefill alone can answer the
            # first token.
            if _left() <= 0:
                log.info("serving probe: %ds budget spent before /v1/completions was attempted", timeout_s)
                return False
            completion = client.post(
                front.rstrip("/") + "/v1/completions",
                json={
                    "model": model_id,
                    "prompt": "hi",
                    "max_tokens": wanted,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": False,
                },
                timeout=_left(),
            )
            if completion.status_code != 200:
                log.info("serving probe: /v1/completions returned %s", completion.status_code)
                return False
            produced = generated_tokens(completion.json())
            if produced < floor:
                log.info("serving probe: /v1/completions generated %d tokens (need %d)", produced, floor)
                return False
    except (httpx.HTTPError, ValueError) as exc:
        log.info("serving probe: %r", exc)
        return False
    return True


def worker_health_targets(state: dict[str, Any] | None, default_port: int) -> list[tuple[str, str, int]]:
    """``(role, ip, port)`` per worker pod; a pod's ``enginePort`` overrides ``default_port``."""
    seen: set[tuple[str, int]] = set()
    targets: list[tuple[str, str, int]] = []
    for role_key in ("prefill_pods", "decode_pods", "worker_pods"):
        for pod in (state or {}).get(role_key) or []:
            if not isinstance(pod, dict):
                continue
            ip = str(pod.get("podIP") or "").strip()
            port = int(pod.get("enginePort") or default_port)
            if not ip or (ip, port) in seen:
                continue
            seen.add((ip, port))
            targets.append((role_key.replace("_pods", ""), ip, port))
    return targets


async def poll_worker_health(
    client: Any, targets: list[tuple[str, str, int]], ready: set[tuple[str, int]], elapsed: int
) -> None:
    """Add each ``(ip, port)`` of ``targets`` whose ``/health`` answers 200 to ``ready``."""
    import httpx

    for role, ip, port in targets:
        if (ip, port) in ready:
            continue
        try:
            resp = await client.get(f"http://{ip}:{port}/health")
        except httpx.HTTPError:
            continue
        if resp.status_code == 200:
            ready.add((ip, port))
            log.info(
                "worker /health OK (%s %s:%d) after %ds [%d/%d]", role, ip, port, elapsed, len(ready), len(targets)
            )
