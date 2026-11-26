# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The narrator: a short LLM-written summary of what the session is doing at the clock, and answers to questions.

The LLM is the one Hyperloom is configured with (``hyperloom_common.llm.llm_config``), imported only when a client is
built, so the rest of the Inspector keeps to the stdlib. Narrations are kept in memory with the clock they describe;
in replay only those at or before the clock are shown or fed back to the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from hyperloom.inspector import digest
from hyperloom.inspector.redact import redact_text

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hyperloom.inspector.watcher import Watcher

log = logging.getLogger(__name__)

MAX_QUESTION = 500
MAX_FOCUS = 300
_TIMEOUT_SEC = 90.0
# The other HTTP timeouts of an LLM call: connecting, sending the request, and waiting for a pooled connection.
_CONNECT_TIMEOUT_SEC = 15.0
_WRITE_TIMEOUT_SEC = 30.0
_POOL_TIMEOUT_SEC = 15.0
# Output tokens per call. Models that think count their thinking against it, so it leaves room for that.
DEFAULT_MAX_TOKENS = 1600
_MAX_ENTRIES = 200
_EARLIER = 5
_REPLAY_BUCKET_SEC = 300.0
# The two LLM backends the narrator can talk to, as HYPERLOOM_INSPECTOR_LLM_BACKEND names them.
_BACKEND_ANTHROPIC = "anthropic"
_BACKEND_OPENAI = "openai"
# The operation of a call that writes a narration; the other, "ask", answers a question.
_OPERATION_NARRATE = "narrate"
# The component every call of the narrator is attributed to. Hyperloom's trace keeps a closed set of LLM components
# (VALID_COMPONENTS in hyperloom.inference_optimizer.trace.llm_trace), which tests/test_contracts_narrator.py holds
# this one to.
_COMPONENT = "inspector"

SYSTEM_NARRATE = """\
You narrate a Hyperloom inference-optimization session for an engineer watching it in the Hyperloom Inspector.
Hyperloom is a multi-agent harness: a coordinator dispatches executor tasks (baseline, roofline, explore, params,
profile, ...) and specialist agents, while LLM roles (orchestration, critic) propose and review changes on a bus.
You get a digest of the session as of one moment: live, or a point in a replay ("now" is the digest's clock).

Write a short status. First a one-line headline. Then at most six bullets, each starting with "- ", covering what is
running now, what just happened, anything that looks wrong (crashes, stalls, errors, memory) and what is likely next.
Use the task, agent and phase names from the digest, and relative times. If an earlier narration is given, say what
changed since. State only what the digest shows; when something isn't visible, say so instead of guessing.
Plain text only, no preamble, no Markdown headings."""

SYSTEM_ASK = """\
You answer questions about a Hyperloom inference-optimization session for an engineer watching it in the Hyperloom
Inspector. Hyperloom is a multi-agent harness: a coordinator dispatches executor tasks and specialist agents, while
LLM roles (orchestration, critic) propose and review changes on a bus. You get a digest of the session as of one
moment ("now" is the digest's clock), plus earlier narrations and answers.

Answer the question from the digest in at most eight short lines; bullets start with "- ". Name the tasks, agents,
phases and times you rely on. If the digest doesn't contain the answer, say so and say what would show it (a log,
an artifact, a tab of the Inspector). Plain text only, no preamble."""

_ENV_FILE_KEY = re.compile(r"^(?:ANTHROPIC_|OPENAI_|HYPERLOOM_INSPECTOR_LLM_)[A-Z0-9_]*$")
_ENV_FILE_EXACT = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_MODEL"}
_HEADER_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _with_cut_note(text: str) -> str:
    """The reply with a note that the model stopped at the token limit, rather than leaving a sentence hanging."""
    return text.rstrip() + "\n[answer cut off at the token limit; raise HYPERLOOM_INSPECTOR_LLM_MAX_TOKENS]"


class Client(Protocol):
    def complete(self, *, system: str, user: str, operation: str) -> str: ...


@dataclass
class _AnthropicClient:
    model: str
    max_tokens: int

    def complete(self, *, system: str, user: str, operation: str) -> str:
        from hyperloom_common.llm import llm_config

        timeout = llm_config.build_http_timeout(
            connect=_CONNECT_TIMEOUT_SEC, read=_TIMEOUT_SEC, write=_WRITE_TIMEOUT_SEC, pool=_POOL_TIMEOUT_SEC
        )
        request = llm_config.AnthropicCompletionRequest(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            timeout=timeout,
            timeout_s=_TIMEOUT_SEC,
        )
        result = llm_config.anthropic_completion(request=request, component=_COMPONENT, operation=operation)
        return _with_cut_note(result.text) if result.stop_reason == "max_tokens" else result.text


@dataclass
class _OpenAIClient:
    client: Any
    model: str
    max_tokens: int

    def complete(self, *, system: str, user: str, operation: str) -> str:
        from hyperloom_common.llm import llm_config

        result = llm_config.chat_completion(
            self.client,
            component=_COMPONENT,
            operation=operation,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return _with_cut_note(result.text) if result.finish_reason == "length" else result.text


@dataclass
class ClientInfo:
    backend: str = "none"
    model: str | None = None
    reason: str | None = None


def build_client(env: Mapping[str, str] | None = None) -> tuple[Client | None, ClientInfo]:
    """The narrator's LLM client from the environment, or None with the reason."""
    env = os.environ if env is None else env
    backend = (env.get("HYPERLOOM_INSPECTOR_LLM_BACKEND") or "auto").strip().lower()
    if backend in ("none", "off", "disabled"):
        return None, ClientInfo(reason="turned off (HYPERLOOM_INSPECTOR_LLM_BACKEND=none)")
    try:
        from hyperloom_common.llm import llm_config
    except ImportError as exc:  # pragma: no cover - the LLM config ships with the package
        return None, ClientInfo(reason=f"Hyperloom's LLM config is not importable: {exc}")
    model = (
        env.get("HYPERLOOM_INSPECTOR_LLM_MODEL") or env.get("CLAUDE_MODEL") or llm_config.DEFAULT_CLAUDE_MODEL
    ).strip()
    try:
        max_tokens = int(env.get("HYPERLOOM_INSPECTOR_LLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    if backend == "auto":
        if llm_config.anthropic_transport_ready(env):
            backend = _BACKEND_ANTHROPIC
        elif env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_KEY"):
            backend = _BACKEND_OPENAI
        else:
            reason = "no LLM credentials in the environment (see --llm-env-file)"
            return None, ClientInfo(reason=reason)
    if backend == _BACKEND_ANTHROPIC:
        if not llm_config.anthropic_transport_ready(env):
            return None, ClientInfo(_BACKEND_ANTHROPIC, model, "no usable Anthropic credential or transport")
        return _AnthropicClient(model, max_tokens), ClientInfo(_BACKEND_ANTHROPIC, model)
    if backend == _BACKEND_OPENAI:
        try:
            timeout = llm_config.build_http_timeout(
                connect=_CONNECT_TIMEOUT_SEC, read=_TIMEOUT_SEC, write=_WRITE_TIMEOUT_SEC, pool=_POOL_TIMEOUT_SEC
            )
            client = llm_config.get_openai_client(timeout=timeout)
        except llm_config.LLMConfigError as exc:
            return None, ClientInfo(_BACKEND_OPENAI, model, f"OpenAI-compatible client unavailable: {exc}")
        return _OpenAIClient(client, model, max_tokens), ClientInfo(_BACKEND_OPENAI, model)
    return None, ClientInfo(reason=f"unknown HYPERLOOM_INSPECTOR_LLM_BACKEND={backend!r}")


def _dotenv(path: Path) -> dict[str, str]:
    """The assignments in a dotenv file, with ``export`` and quotes removed.

    A quoted value may span lines up to its closing quote, and text after that quote (a comment) is dropped.
    """
    values = {}
    lines = iter(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
    for raw in lines:
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value[:1] in ("'", '"'):
            quote, body = value[0], value[1:]
            while quote not in body:
                more = next(lines, None)
                if more is None:
                    break
                body += "\n" + more
            value = body.partition(quote)[0]
        else:
            value = value.partition(" #")[0].rstrip()
        values[key] = value
    return values


def load_llm_env(path: Path, environ: dict[str, str] | None = None) -> list[str]:
    """Set the LLM variables *path* (a dotenv file) has and the environment lacks; returns their names.

    Only ``ANTHROPIC_*``, ``OPENAI_*``, ``HYPERLOOM_INSPECTOR_LLM_*``, ``CLAUDE_CODE_OAUTH_TOKEN`` and
    ``CLAUDE_MODEL`` are read, plus the variables a ``*_CUSTOM_HEADERS`` value refers to as ``${NAME}`` (a gateway's
    subscription header is often kept that way); values are never logged.
    """
    environ = os.environ if environ is None else environ  # type: ignore[assignment]
    values = _dotenv(path)
    wanted = {k for k in values if _ENV_FILE_KEY.match(k) or k in _ENV_FILE_EXACT}
    for key in [k for k in wanted if k.endswith("_CUSTOM_HEADERS")]:
        wanted.update(_HEADER_REF.findall(environ.get(key) or values[key]))
    loaded = []
    for key in sorted(wanted):
        if key not in environ and values.get(key):
            environ[key] = values[key]
            loaded.append(key)
    return loaded


@dataclass
class _Gate:
    """One LLM call at a time, and at least *interval* seconds between generated narrations."""

    interval: float
    busy: bool = False
    last: float = -1e18
    lock: threading.Lock = field(default_factory=threading.Lock)


class Narrator:
    """Narrations and answers for one watcher; thread-safe."""

    def __init__(
        self,
        watcher: Watcher,
        client: Client | None,
        info: ClientInfo,
        min_interval: float = 20.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.watcher = watcher
        self.client = client
        self.info = info
        self.entries: list[dict] = []
        self._gate = _Gate(min_interval)
        self._monotonic = monotonic

    # ----- reads ---------------------------------------------------------------------------------------------

    def _visible(self, clock: float) -> list[dict]:
        """Entries at or before *clock* (all of them live), oldest first."""
        if not self.watcher.clock.replay:
            return list(self.entries)
        return [e for e in self.entries if e["clock"] <= clock + 1e-6]

    def status(self) -> dict:
        clock = self.watcher.clock.now()
        with self._gate.lock:
            busy, entries = self._gate.busy, self._visible(clock)
        return {
            "available": self.client is not None,
            "reason": self.info.reason,
            "backend": self.info.backend,
            "model": self.info.model,
            "busy": busy,
            "clock": clock,
            "entries": entries[-50:],
        }

    # ----- writes --------------------------------------------------------------------------------------------

    def narrate(self, force: bool = False, focus: str = "") -> tuple[int, dict]:
        """A narration at the clock: the latest one when nothing material changed (unless *force*), else a new one."""
        if self.client is None:
            return HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"narrator unavailable: {self.info.reason}"}
        focus = _clean(focus, MAX_FOCUS)
        view = self.watcher.snapshot()
        clock = view["clock"]["now"]
        fp = _fingerprint(view, focus, self.watcher.clock.replay)
        with self._gate.lock:
            earlier = self._visible(clock)
            latest = next((e for e in reversed(earlier) if e["kind"] == "narration"), None)
            if latest is not None and latest.get("fingerprint") == fp and not force:
                return HTTPStatus.OK, {"entry": latest, "reused": True}
            refusal = self._claim(interval=True)
            if refusal:
                return HTTPStatus.TOO_MANY_REQUESTS, {"error": refusal, "entry": latest}
        prompt = digest.build(view, earlier[-_EARLIER:])
        if focus:
            prompt += f"\n\n## The user's focus\nPay particular attention to: {focus}"
        return self._call(_OPERATION_NARRATE, SYSTEM_NARRATE, prompt, clock, fingerprint=fp, focus=focus or None)

    def ask(self, question: str, focus: str = "") -> tuple[int, dict]:
        """An answer to *question*, from the digest at the clock and the earlier narrations and answers."""
        if self.client is None:
            return HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"narrator unavailable: {self.info.reason}"}
        question, focus = _clean(question, MAX_QUESTION), _clean(focus, MAX_FOCUS)
        if not question:
            return HTTPStatus.BAD_REQUEST, {"error": "empty question"}
        view = self.watcher.snapshot()
        clock = view["clock"]["now"]
        with self._gate.lock:
            earlier = self._visible(clock)
            refusal = self._claim(interval=False)
            if refusal:
                return HTTPStatus.TOO_MANY_REQUESTS, {"error": refusal}
        prompt = digest.build(view, earlier[-_EARLIER:])
        if focus:
            prompt += f"\n\n## The user's focus\n{focus}"
        prompt += f"\n\n## Question\n{question}"
        return self._call("ask", SYSTEM_ASK, prompt, clock, question=question, focus=focus or None)

    def _claim(self, interval: bool) -> str | None:
        """Take the gate (its lock held), or say why not."""
        gate, now = self._gate, self._monotonic()
        if gate.busy:
            return "the narrator is already writing; try again in a moment"
        if interval and now - gate.last < gate.interval:
            return f"a narration was written {now - gate.last:.0f}s ago; the minimum gap is {gate.interval:.0f}s"
        gate.busy = True
        return None

    def _call(self, operation: str, system: str, prompt: str, clock: float, **extra: Any) -> tuple[int, dict]:
        assert self.client is not None
        started = time.time()
        try:
            text = self.client.complete(system=system, user=prompt, operation=operation)
        except Exception as exc:  # noqa: BLE001 - any client failure becomes a message in the panel
            log.warning("inspector narrator: %s failed: %s", operation, redact_text(str(exc)))
            return HTTPStatus.BAD_GATEWAY, {"error": redact_text(f"{type(exc).__name__}: {exc}")[:500]}
        finally:
            with self._gate.lock:
                self._gate.busy = False
                if operation == _OPERATION_NARRATE:
                    self._gate.last = self._monotonic()
        entry = {
            "id": uuid.uuid4().hex[:12],
            "kind": "narration" if operation == _OPERATION_NARRATE else "answer",
            "clock": clock,
            "written_at": started,
            "duration_sec": round(time.time() - started, 1),
            "model": self.info.model,
            "text": redact_text((text or "").strip()) or "(the model returned no text)",
            **{k: v for k, v in extra.items() if v is not None},
        }
        with self._gate.lock:
            self.entries.append(entry)
            del self.entries[:-_MAX_ENTRIES]
        return HTTPStatus.OK, {"entry": entry, "reused": False}


def _clean(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _fingerprint(view: dict, focus: str, replay: bool) -> str:
    """What makes a new narration worth writing: phase, released records, task states, focus, and in replay the
    clock to the nearest few minutes (so a paused scrub to a new spot is narrated anew)."""
    released = {k: v for k, v in (view.get("released") or {}).items() if k != "metrics"}
    tasks = sorted((t.get("task_id"), t.get("state"), t.get("health")) for t in view.get("tasks") or [])
    parts: list[Any] = [(view.get("summary") or {}).get("phase"), released, tasks, focus]
    if replay:
        parts.append(int((view.get("clock") or {}).get("now", 0) // _REPLAY_BUCKET_SEC))
    return hashlib.sha256(json.dumps(parts, default=str, sort_keys=True).encode()).hexdigest()[:16]
