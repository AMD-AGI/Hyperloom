"""Single-shot checks for whether a multi-node cluster is actually serving.

The post-restart readiness *wait* lives in the orchestrator; these are its
one-attempt counterparts, for deciding whether a resume may skip a relaunch.
Same questions, asked once and answered now rather than polled to a deadline.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .external_state import reachable_service_url

log = logging.getLogger(__name__)

# Mirror the orchestrator's post-restart probe so a resume and the wait that
# follows it agree on what "serving" means.
_PROBE_TOKENS_ENV = "HYPERLOOM_MN_COMPLETION_PROBE_TOKENS"
_PROBE_MIN_TOKENS_ENV = "HYPERLOOM_MN_COMPLETION_PROBE_MIN_TOKENS"
_DEFAULT_PROBE_TOKENS = 8
_DEFAULT_PROBE_MIN_TOKENS = 2


def _int_env(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on junk.

    Args:
        name: Environment variable to read.
        default: Value used when unset or unparseable.

    Returns:
        int: The resolved value.
    """
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def generated_tokens(data: object) -> int:
    """Best-effort count of tokens a ``/v1/completions`` probe actually produced.

    Prefers ``usage.completion_tokens``; falls back to a non-empty
    ``choices[0].text`` (counts as 1). Returns 0 when nothing was generated --
    a broken PD KV handoff answers HTTP 200 with an empty completion, so a
    status-only check would call a decode leg ready while it serves nothing.

    Args:
        data: Parsed JSON body of a ``/v1/completions`` response.

    Returns:
        int: Generated token count, 0 when none or unparseable.
    """
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
    """The endpoints that must each be up before the cluster counts as serving.

    Disaggregation splits serving across a prefill and a decode group, and the
    frontend answers while either is still loading, so both legs are addressed
    directly. Their URLs come from the launcher's own summary, recorded on the
    last successful restart. Aggregated runs serve from rank 0 alone, which only
    binds once the whole distributed group has joined.

    Args:
        state: Multi-node state.
        pd_mode: ``"disaggregated"`` or anything else.

    Returns:
        list[str]: Base URLs to health-check, empty when they cannot be resolved
        (which callers must treat as "cannot verify", not as "healthy").
    """
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

    Three questions, because each catches what the previous one misses. Every
    leg's ``/health`` covers a group that is only half up. ``/v1/models``
    non-empty covers workers that died during the weight load, since the
    frontend serves its own health long before any model registers. A short
    ``/v1/completions`` covers the case both of those pass and nothing is
    generated anyway -- a PD KV handoff that returns 200 with an empty
    completion.

    A False answer does NOT mean the cluster is dead: a cold start looks
    identical from out here. Callers deciding whether to relaunch must
    distinguish those separately.

    Args:
        state: Multi-node state naming the frontend and, for PD, both legs.
        pd_mode: ``"disaggregated"`` or anything else.
        timeout_s: Budget for the whole probe, not per request. Each call gets
            what is left of it, so a slow leg cannot hand the next check a fresh
            budget. httpx applies a timeout per phase rather than per call, so a
            single request can still overrun by one connect phase.

    Returns:
        bool: True only when every check passes on this attempt.
    """
    legs = _serving_legs(state, pd_mode)
    front = reachable_service_url(state)
    if not legs or not front:
        return False
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is backfilled by preflight
        log.info("serving probe: httpx unavailable")
        return False

    deadline = time.monotonic() + timeout_s

    def _left() -> float:
        """Seconds still available to the probe.

        Returns:
            float: Remaining budget; <= 0 once it is spent.
        """
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

            # ignore_eos plus several tokens forces the request through the
            # decode leg; prefill alone can answer the first token.
            wanted = _int_env(_PROBE_TOKENS_ENV, _DEFAULT_PROBE_TOKENS)
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
            floor = _int_env(_PROBE_MIN_TOKENS_ENV, _DEFAULT_PROBE_MIN_TOKENS)
            if produced < floor:
                log.info("serving probe: /v1/completions generated %d tokens (need %d)", produced, floor)
                return False
    except Exception as exc:  # noqa: BLE001
        log.info("serving probe: %r", exc)
        return False
    return True
