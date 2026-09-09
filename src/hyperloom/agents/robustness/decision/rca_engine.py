# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCA engines, all exposing ``async def summarize(symptom) -> str``."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from hyperloom.common import llm_config

from ..signals import Symptom, SymptomSeverity

if TYPE_CHECKING:
    from ..state_store import DetectorStateView


log = logging.getLogger(__name__)


@runtime_checkable
class RcaEngine(Protocol):
    """Minimal contract the ActionLadder consumes."""

    async def summarize(self, symptom: Symptom) -> str:
        """Produce root-cause text for a symptom."""

    async def aclose(self) -> None:
        """Release any provider client the engine owns."""


@dataclass
class NoopRcaEngine:
    """Default engine: emits no RCA text."""

    label: str = "noop"

    async def summarize(self, symptom: Symptom) -> str:
        """Return empty RCA text; this engine never contacts an LLM."""
        return ""

    def drain_usage(self) -> dict[str, Any] | None:
        """No LLM is ever contacted, so there is never any usage to drain."""
        return None

    async def aclose(self) -> None:
        """No client is ever created, so there is nothing to close."""


@dataclass
class RcaThrottleConfig:
    """Tunables that bound LLM RCA cost."""

    severity_min: SymptomSeverity = SymptomSeverity.HIGH
    cooldown_seconds: float = 60.0
    max_calls_per_tick: int = 1


class RcaThrottle:
    """Tick-aware cost guard for LLM RCA calls."""

    def __init__(
        self,
        config: RcaThrottleConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the throttle and load any persisted cooldown state."""
        self._config = config or RcaThrottleConfig()
        self._state_view = state_view
        # Disk-backed per-key cooldown timestamps; per-tick counters stay in-memory.
        loaded = state_view.load() if state_view is not None else {}
        self._last_called_unix: dict[tuple[str, ...], float] = _decode_throttle_keys(loaded.get("last_called_unix"))
        self._tick_calls = 0
        self._tick_id: int | None = None

    @property
    def config(self) -> RcaThrottleConfig:
        """Return the active throttle configuration."""
        return self._config

    def _persist(self) -> None:
        """Write the current cooldown timestamps to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save(
            {
                "last_called_unix": _encode_throttle_keys(self._last_called_unix),
            }
        )

    def begin_tick(self, tick_id: int) -> None:
        """Reset the per-tick call counter when a new tick begins."""
        if self._tick_id != tick_id:
            self._tick_id = tick_id
            self._tick_calls = 0

    def should_call(self, sym: Symptom, *, now_unix: float, tick_id: int) -> bool:
        """Decide whether an LLM call is permitted for this symptom now."""
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
        """Record that an LLM call was made for a symptom and persist it."""
        self._last_called_unix[sym.dedup_key()] = now_unix
        self._tick_calls += 1
        self._persist()


_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rca.md"


@lru_cache(maxsize=1)
def load_rca_system_prompt() -> str:
    """Read the RCA system prompt shipped as package data."""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# Floor for the Claude CLI transport's wall-clock budget.
_CLI_MIN_TIMEOUT_SEC = 60.0


def _client_timeout(timeout_s: float) -> Any:
    """Build the uniform per-request timeout both engines run with."""
    return llm_config.build_http_timeout(connect=timeout_s, read=timeout_s, write=timeout_s, pool=timeout_s)


def _provider_env(*, api_key_env: str, api_key: str, base_url_env: str, base_url: str) -> dict[str, str]:
    """Overlay the discovered RCA credentials onto the process env."""
    return {**os.environ, api_key_env: api_key, base_url_env: base_url}


@dataclass
class LlmRcaEngine:
    """Async OpenAI-compatible RCA engine (chat-server proxy)."""

    base_url: str
    api_key: str
    model: str = "claude-opus-5"
    timeout_s: float = 8.0
    max_chars: int = 1500
    throttle: RcaThrottle | None = None
    # Provider client; built on first use unless a caller injects one.
    client: Any = None
    _owns_client: bool = field(default=False, init=False, repr=False)
    _config_warned: bool = field(default=False, init=False, repr=False)
    # Set once a failure proves further calls cannot succeed, so a permanent misconfiguration costs one ERROR rather
    # than a warning every tick.
    _disabled: bool = field(default=False, init=False, repr=False)
    _current_tick_id: int = field(default=-1, init=False, repr=False)
    # Token-usage accumulator across the calls made since the last drain, so the host (Coordinator) can fold the RCA
    # LLM spend into its trace ledger.
    _usage_in: int = field(default=0, init=False, repr=False)
    _usage_out: int = field(default=0, init=False, repr=False)
    _usage_calls: int = field(default=0, init=False, repr=False)
    _usage_latency_ms: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Warn about missing credentials and install the default throttle."""
        if not self._is_configured():
            log.warning("%s constructed without base_url/api_key; calls will be skipped", type(self).__name__)
        if self.throttle is None:
            self.throttle = RcaThrottle()

    def _is_configured(self) -> bool:
        """Whether this engine holds everything a call needs."""
        return bool(self.base_url and self.api_key)

    def _new_client(self) -> Any:
        """Build the OpenAI-compatible client this engine calls through."""
        return llm_config.get_async_openai_client(
            env=_provider_env(
                api_key_env="OPENAI_API_KEY",
                api_key=self.api_key,
                base_url_env="OPENAI_BASE_URL",
                base_url=self.base_url,
            ),
            timeout=_client_timeout(self.timeout_s),
        )

    def _ensure_client(self) -> Any:
        """Return the provider client, building it on first use."""
        if self.client is None:
            self.client = self._new_client()
            self._owns_client = True
        return self.client

    async def _aclose_client(self) -> None:
        """Close the OpenAI SDK client."""
        await self.client.close()

    async def aclose(self) -> None:
        """Close the provider client if this engine created it."""
        if self._owns_client and self.client is not None:
            await self._aclose_client()

    def drain_usage(self) -> dict[str, Any] | None:
        """Return + reset the token usage accumulated since the last drain."""
        if self._usage_calls <= 0:
            return None
        out: dict[str, Any] = {
            "input_tokens": self._usage_in,
            "output_tokens": self._usage_out,
            "calls": self._usage_calls,
            "latency_ms": self._usage_latency_ms,
            "model": self.model,
        }
        self._usage_in = 0
        self._usage_out = 0
        self._usage_calls = 0
        self._usage_latency_ms = 0
        return out

    async def summarize(self, symptom: Symptom) -> str:
        """Summarize a symptom via the chat-server, subject to throttling."""
        if self._disabled or not self._is_configured():
            return ""
        now_unix = time.time()
        # tick_id = -1 = single shared bucket when no caller sets one; ActionLadder scopes per-tick buckets via
        # set_tick (see decide()).
        tick_id = self._current_tick_id
        assert self.throttle is not None
        if not self.throttle.should_call(symptom, now_unix=now_unix, tick_id=tick_id):
            return ""

        text = await self._call(symptom)
        self.throttle.record(symptom, now_unix=now_unix)
        return _truncate(text, self.max_chars)

    def _note_call_failure(self, exc: BaseException) -> None:
        """Record a failed call, disabling the engine when it can only recur."""
        permanent = isinstance(exc, llm_config.LLMConfigError) or not self._is_configured()
        if permanent:
            self._disabled = True
            log.error("%s disabled; RCA cannot run in this configuration: %r", type(self).__name__, exc)
            return
        log.warning("%s: completion failed: %r", type(self).__name__, exc)

    def _accumulate_usage(self, usage: Any, *, latency_ms: int) -> None:
        """Fold one chat response's ``usage`` object into the accumulator."""
        self._usage_calls += 1
        self._usage_latency_ms += max(0, int(latency_ms))
        if usage is None:
            return
        try:
            self._usage_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            self._usage_out += int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass

    def set_tick(self, tick_id: int) -> None:
        """Hook used by ActionLadder to scope per-tick budgets."""
        self._current_tick_id = tick_id
        if self.throttle is not None:
            self.throttle.begin_tick(tick_id)

    async def _call(self, symptom: Symptom) -> str:
        """Issue the chat-completion request and extract the reply text."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": load_rca_system_prompt()},
                {"role": "user", "content": _build_user_prompt(symptom)},
            ],
        }
        if _uses_max_completion_tokens(self.model):
            params["max_completion_tokens"] = 600
        else:
            params["max_tokens"] = 600
            params["temperature"] = 0.2
        _t0 = time.perf_counter()
        try:
            result = await llm_config.achat_completion(
                self._ensure_client(),
                component="robustness",
                operation="analyze_symptom",
                **params,
            )
        except Exception as exc:  # noqa: BLE001 - degrade-to-empty is this engine's contract
            self._note_call_failure(exc)
            return ""
        self._accumulate_usage(result.usage, latency_ms=int((time.perf_counter() - _t0) * 1000))
        return str(result.text or "").strip()


@dataclass
class AnthropicRcaEngine(LlmRcaEngine):
    """Anthropic-side RCA engine, issuing one single-shot completion."""

    def _is_configured(self) -> bool:
        """Whether a completion could actually be issued."""
        return llm_config.anthropic_transport_ready(self._resolved_env())

    def _resolved_env(self) -> dict[str, str]:
        """Environment carrying whatever ``Config.discover`` resolved."""
        return _provider_env(
            api_key_env="ANTHROPIC_API_KEY",
            api_key=self.api_key,
            base_url_env="ANTHROPIC_BASE_URL",
            base_url=self.base_url,
        )

    async def _call(self, symptom: Symptom) -> str:
        """Issue one tool-free Anthropic completion and extract text content."""
        _t0 = time.perf_counter()
        try:
            result = await llm_config.aanthropic_completion(
                component="robustness",
                operation="analyze_symptom",
                model=self.model,
                system=load_rca_system_prompt(),
                messages=[{"role": "user", "content": _build_user_prompt(symptom)}],
                max_tokens=600,
                temperature=0.2,
                env=self._resolved_env(),
                timeout=_client_timeout(self.timeout_s),
                timeout_s=max(self.timeout_s, _CLI_MIN_TIMEOUT_SEC),
            )
        except Exception as exc:  # noqa: BLE001 - degrade-to-empty is this engine's contract
            self._note_call_failure(exc)
            return ""
        self._accumulate_anthropic_usage(result.usage, latency_ms=int((time.perf_counter() - _t0) * 1000))
        return str(result.text or "").strip()

    def _accumulate_anthropic_usage(self, usage: Any, *, latency_ms: int) -> None:
        """Fold Anthropic ``usage`` fields into the shared accumulator."""
        self._usage_calls += 1
        self._usage_latency_ms += max(0, int(latency_ms))
        if not isinstance(usage, Mapping):
            return
        try:
            self._usage_in += int(usage.get("input_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            self._usage_out += int(usage.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass


def _build_user_prompt(sym: Symptom) -> str:
    """Render a symptom into a prompt string."""
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
    return "\n".join(lines)


def _uses_max_completion_tokens(model: str) -> bool:
    """Return whether an OpenAI-compatible model rejects legacy max_tokens."""
    return str(model or "").strip().lower().startswith("gpt-5")


def _format_evidence(payload: Any, prefix: str = "  ") -> list[str]:
    """Flatten arbitrary evidence into indented, human-readable lines."""
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


def _truncate(text: str, max_chars: int) -> str:
    """Trim text to a maximum length, appending an ellipsis when cut."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ASCII unit separator; keeps tuple keys round-trippable through JSON object keys.
_THROTTLE_KEY_SEP: str = "\x1f"


def _encode_throttle_keys(
    last_called: dict[tuple[str, ...], float],
) -> dict[str, float]:
    """Serialise tuple-keyed cooldown timestamps to a JSON-safe dict."""
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
    """Inverse of :func:`_encode_throttle_keys`; tolerant of bad input."""
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
    "AnthropicRcaEngine",
    "LlmRcaEngine",
    "NoopRcaEngine",
    "RcaEngine",
    "RcaThrottle",
    "RcaThrottleConfig",
    "load_rca_system_prompt",
]
