# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""RCA engines.

M1.5 ships two engines:

* :class:`NoopRcaEngine` — default. Returns "" so the ActionLadder skips
  the ``rca_text`` field on findings.
* :class:`LlmRcaEngine` — calls an OpenAI-compatible chat completion
  endpoint (the chat-server proxying Codex / Claude). The engine is
  guarded by :class:`RcaThrottle` so cost stays bounded:

    - severity gate (default: only ``high`` symptoms),
    - per-dedup-key cooldown (default 60s),
    - per-tick cap (default 1 LLM call).

Both engines expose ``async def summarize(symptom) -> str``. The
ActionLadder ``await``s each call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import httpx

from ..signals import Symptom, SymptomSeverity
from ..state_store import DetectorStateView


log = logging.getLogger(__name__)


@runtime_checkable
class RcaEngine(Protocol):
    """Minimal contract the ActionLadder consumes."""

    async def summarize(self, symptom: Symptom) -> str:
        ...


@dataclass
class NoopRcaEngine:
    """Default engine: emits no RCA text."""

    label: str = "noop"

    async def summarize(self, symptom: Symptom) -> str:
        return ""


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

@dataclass
class RcaThrottleConfig:
    severity_min: SymptomSeverity = SymptomSeverity.HIGH
    cooldown_seconds: float = 60.0
    max_calls_per_tick: int = 1


class RcaThrottle:
    """Tick-aware cost guard for LLM RCA calls.

    The ActionLadder/Reactor calls :meth:`begin_tick` once per tick (the
    LlmRcaEngine does it lazily on the first ``summarize`` of a tick).
    :meth:`should_call` then both checks the budget and returns whether
    the engine should actually contact the LLM.
    """

    def __init__(
        self,
        config: RcaThrottleConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or RcaThrottleConfig()
        self._state_view = state_view
        # Disk-backed per-key cooldown timestamps — the 60s default
        # cooldown is meaningless without persistence under the
        # subprocess-per-tick transport (every subprocess sees an
        # empty dict and calls the LLM again). ``_tick_calls`` /
        # ``_tick_id`` deliberately stay in-memory: they're per-tick
        # budget counters, not cross-tick state.
        loaded = state_view.load() if state_view is not None else {}
        self._last_called_unix: dict[tuple[str, ...], float] = (
            _decode_throttle_keys(loaded.get("last_called_unix"))
        )
        self._tick_calls = 0
        self._tick_id: int | None = None

    @property
    def config(self) -> RcaThrottleConfig:
        return self._config

    def _persist(self) -> None:
        if self._state_view is None:
            return
        self._state_view.save({
            "last_called_unix": _encode_throttle_keys(
                self._last_called_unix
            ),
        })

    def begin_tick(self, tick_id: int) -> None:
        if self._tick_id != tick_id:
            self._tick_id = tick_id
            self._tick_calls = 0

    def should_call(self, sym: Symptom, *, now_unix: float, tick_id: int) -> bool:
        self.begin_tick(tick_id)
        if sym.severity.rank < self._config.severity_min.rank:
            return False
        if self._tick_calls >= self._config.max_calls_per_tick:
            return False
        last = self._last_called_unix.get(sym.dedup_key())
        if last is not None and (now_unix - last) < self._config.cooldown_seconds:
            return False
        return True

    def record(self, sym: Symptom, *, now_unix: float) -> None:
        self._last_called_unix[sym.dedup_key()] = now_unix
        self._tick_calls += 1
        self._persist()


# ---------------------------------------------------------------------------
# LLM engine
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Hyperloom robustness reactor RCA assistant. Given one symptom and \
its evidence, write a concise root-cause summary in <= 6 sentences. \
Focus on actionable remediation hints and observable evidence. If the \
evidence is insufficient, reply exactly with: insufficient evidence.
"""


@dataclass
class LlmRcaEngine:
    """Async OpenAI-compatible RCA engine (chat-server proxy).

    The HTTP layer goes directly through ``httpx.AsyncClient`` rather
    than the ``openai`` SDK to keep error handling explicit and
    mocking trivial. ``base_url`` should already include any version
    prefix (eg. ``/v1``).
    """

    base_url: str
    api_key: str
    model: str = "claude-opus-4-7"
    timeout_s: float = 8.0
    max_chars: int = 1500
    throttle: RcaThrottle | None = None
    client: httpx.AsyncClient | None = None
    extra_evidence_provider: Any | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)
    _config_warned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url or not self.api_key:
            log.warning("LlmRcaEngine constructed without base_url/api_key; calls will be skipped")
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=httpx.Timeout(self.timeout_s),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._owns_client = True
        if self.throttle is None:
            self.throttle = RcaThrottle()

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    async def summarize(self, symptom: Symptom) -> str:
        if not self.base_url or not self.api_key:
            return ""
        now_unix = time.time()
        # tick_id = -1 routes per-tick budget to a single bucket when
        # callers don't provide one; ActionLadder calls begin_tick via
        # its own wrapper to scope buckets per tick (see decide()).
        tick_id = getattr(self, "_current_tick_id", -1)
        assert self.throttle is not None
        if not self.throttle.should_call(symptom, now_unix=now_unix, tick_id=tick_id):
            return ""

        text = await self._call(symptom)
        self.throttle.record(symptom, now_unix=now_unix)
        return _truncate(text, self.max_chars)

    def set_tick(self, tick_id: int) -> None:
        """Hook used by ActionLadder to scope per-tick budgets."""
        self._current_tick_id = tick_id
        if self.throttle is not None:
            self.throttle.begin_tick(tick_id)

    async def _call(self, symptom: Symptom) -> str:
        prompt = _build_user_prompt(symptom, self.extra_evidence_provider)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.2,
        }
        try:
            assert self.client is not None
            resp = await self.client.post("/chat/completions", json=payload)
        except httpx.TimeoutException:
            log.warning("LlmRcaEngine: chat-server call timed out")
            return ""
        except httpx.RequestError as exc:
            log.warning("LlmRcaEngine: chat-server request failed: %s", exc)
            return ""
        if resp.status_code >= 400:
            log.warning(
                "LlmRcaEngine: chat-server status=%d body=%s",
                resp.status_code,
                resp.text[:200],
            )
            return ""
        try:
            body = resp.json()
        except ValueError:
            log.warning("LlmRcaEngine: chat-server returned non-json body")
            return ""
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            return ""
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, list):
            # Some providers return a list of content parts
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        return str(content or "").strip()


def _build_user_prompt(
    sym: Symptom,
    extra_evidence_provider: Any | None,
) -> str:
    lines = [
        f"symptom: {sym.name}",
        f"severity: {sym.severity.value}",
        f"summary: {sym.summary}",
    ]
    if sym.subject:
        lines.append("subject:")
        for k, v in sorted(sym.subject.items()):
            lines.append(f"  {k}={v}")
    if sym.evidence:
        lines.append("evidence:")
        lines.extend(_format_evidence(sym.evidence))
    if sym.suggestion:
        lines.append(f"suggestion_hint: {sym.suggestion}")
    extra = _safe_extra_evidence(extra_evidence_provider, sym)
    if extra:
        lines.append("recent_log_errors:")
        for entry in extra[:5]:
            lines.append(f"  - {entry}")
    return "\n".join(lines)


def _format_evidence(payload: Any, prefix: str = "  ") -> list[str]:
    if isinstance(payload, Mapping):
        out: list[str] = []
        for k in sorted(payload.keys()):
            v = payload[k]
            if isinstance(v, (str, int, float, bool)) or v is None:
                out.append(f"{prefix}{k}: {v}")
            else:
                out.append(f"{prefix}{k}:")
                out.extend(_format_evidence(v, prefix + "  "))
        return out
    if isinstance(payload, (list, tuple)):
        return [f"{prefix}- {item}" for item in payload[:10]]
    return [f"{prefix}{payload}"]


def _safe_extra_evidence(provider: Any | None, sym: Symptom) -> list[str]:
    if provider is None:
        return []
    try:
        items = provider(sym)
    except Exception:
        log.exception("rca extra evidence provider failed")
        return []
    if not isinstance(items, list):
        return []
    return [str(it)[:240] for it in items][:10]


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Throttle state (de)serialisation helpers
# ---------------------------------------------------------------------------

# ASCII unit separator — same scheme as the ActionLadder cooldown
# encoder; keeps tuple keys round-trippable through JSON object keys.
_THROTTLE_KEY_SEP: str = "\x1f"


def _encode_throttle_keys(
    last_called: dict[tuple[str, ...], float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, ts in last_called.items():
        try:
            encoded = _THROTTLE_KEY_SEP.join(str(part) for part in key)
        except Exception:  # noqa: BLE001
            continue
        try:
            out[encoded] = float(ts)
        except (TypeError, ValueError):
            continue
    return out


def _decode_throttle_keys(payload: Any) -> dict[tuple[str, ...], float]:
    if not isinstance(payload, dict):
        return {}
    out: dict[tuple[str, ...], float] = {}
    for raw_key, raw_ts in payload.items():
        if not isinstance(raw_key, str):
            continue
        try:
            ts = float(raw_ts)
        except (TypeError, ValueError):
            continue
        parts = tuple(raw_key.split(_THROTTLE_KEY_SEP))
        out[parts] = ts
    return out


__all__ = [
    "LlmRcaEngine",
    "NoopRcaEngine",
    "RcaEngine",
    "RcaThrottle",
    "RcaThrottleConfig",
]
