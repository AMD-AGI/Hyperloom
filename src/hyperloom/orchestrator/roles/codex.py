# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CodexBackend — a reactor role driven by the Codex Agent SDK.

Hyperloom issues no bare LLM API calls: every agent runs inside an agent
runtime, so auth, sandboxing, turn timeouts and usage accounting have exactly
one owner. This backend is the OpenAI-side half of that contract for the
Coordinator, the mirror of :class:`ClaudeBackend` on the Anthropic side.

The intent transport is a provider-enforced structured output rather than
prose the model is asked to imitate. Each turn ships the JSON schema
:func:`build_intent_envelope_schema` renders from the *role's own* intent set,
so the legal ``intent_type`` values are the role record's — never a literal
that can fall behind it. Strict structured outputs cannot express a free-form
object, so the payload arrives as a JSON string; decoding it and handing the
result to :func:`validate_envelope` keeps payload checking identical to the
Claude path.

One :class:`CodexSession` is held across ticks, which is what makes the
backend honestly ``conversational``: the Coordinator can then push a thin
delta instead of the full session state every turn, and compaction has a
conversation to compact. The session owns a child process and a private state
directory, so the Coordinator closes it (:meth:`aclose`) when the run ends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.codex_session import (
    CodexSession,
    CodexSessionError,
    CodexSessionUnavailableError,
)
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from ..prompts.transport import TRANSPORT_STRUCTURED_OUTPUT
from .agent_role import DEFAULT_CODEX_MODEL
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    RetryPolicy,
    parse_call_timeout_env,
    retry_with_backoff,
    safe_int,
)
from .mcp_emit_intent import (
    build_intent_envelope_schema,
    constraints_sentence,
    payload_contract,
)

# Prepended to a turn that runs without the enforced schema. Thread developer
# instructions are fixed at thread_start, so the standing OUTPUT FORMAT block
# cannot be scoped out for one turn and has to be overridden in the user turn.
_PER_TURN_SCHEMA_OVERRIDE = (
    "THIS TURN ONLY: ignore the OUTPUT FORMAT block in your instructions. "
    "Do not emit an intent envelope. Reply exactly as the message below asks."
)


def build_output_instructions(allowed_intents: Iterable[IntentType]) -> str:
    """Render the transport contract for one role's intent set.

    The schema is enforced by the provider, so this text only has to explain
    the two things a schema cannot: that the payload is a serialized object,
    and that a turn with nothing to report still owes an intent. It is scoped
    to ``allowed_intents`` so no role is ever handed another role's contract.

    Args:
        allowed_intents: The intent types the role may emit.

    Returns:
        The output-format block appended to the thread's developer
        instructions.
    """
    contract = payload_contract(allowed_intents)
    constraints = constraints_sentence(allowed_intents)
    constraints_line = f"\n-{constraints}" if constraints else ""
    heartbeat = json.dumps({"topic": "heartbeat", "body_md": "ok"})
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
Your final message MUST be exactly one JSON object matching the enforced
output schema — no prose, no code fences, nothing around it:

{{"intents": [{{"intent_type": "...", "payload": "..."}}]}}

- `payload` is a STRING holding a serialized JSON object. The schema cannot
  express a free-form object, so serialize the payload and escape it.
- Required keys per intent_type: {contract}.{constraints_line}
- Emit several intents by adding entries to `intents`.
- Put only NEW information in payload bodies; do not restate context already
  in SharedState, your inbox, or analysis.md. Keep length proportional to
  substance.
- ALWAYS emit at least one intent. With nothing to report, emit
  {{"intent_type": "send_message", "payload": {json.dumps(heartbeat)}}}.
==== END OUTPUT FORMAT ====
""".strip()


def decode_intent_envelope(text: str) -> dict[str, Any]:
    """Decode a schema-enforced reply into a :func:`validate_envelope` input.

    Inflates every JSON-string payload back into the object the shared
    validator expects. The reply shape is provider-enforced, so anything
    unparseable here means the structured-output constraint did not hold and
    the turn produced nothing usable.

    Args:
        text: The model's final response.

    Returns:
        The envelope with object payloads.

    Raises:
        NoIntentEmitted: If the reply or any payload is not decodable JSON of
            the expected shape.
    """
    try:
        envelope = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise NoIntentEmitted(
            f"codex reply did not honour the enforced output schema (chars={len(text or '')})"
        ) from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("intents"), list):
        raise NoIntentEmitted("codex reply is valid JSON but carries no 'intents' list")
    # An empty list satisfies validate_envelope, so without this the tick is
    # recorded as a success that did nothing and the backend's error streak is
    # reset. The schema's ``minItems`` is the provider-side belt; this is the
    # braces, because a gateway that drops the keyword must not go unnoticed.
    if not envelope["intents"]:
        raise NoIntentEmitted("codex reply carried an empty 'intents' list")
    decoded: list[Any] = []
    for index, item in enumerate(envelope["intents"]):
        if not isinstance(item, dict):
            raise NoIntentEmitted(f"codex intents[{index}] is not an object")
        payload = item.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError as exc:
                raise NoIntentEmitted(f"codex intents[{index}] payload is not decodable JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise NoIntentEmitted(f"codex intents[{index}] payload did not decode to an object")
        decoded.append({"intent_type": item.get("intent_type"), "payload": payload})
    return {"intents": decoded}


@dataclass
class CodexBackend:
    """Production Codex reactor backend. Implements :class:`Backend`.

    One behaviour differs from the Claude path and cannot be made to match. A
    thread's developer instructions are fixed when it starts, so a phase re-scope
    has to open a new thread, and the model's own reasoning and plan go with the
    old one. :meth:`needs_seed_for` makes the Coordinator push a full SEED across
    that seam, and the seam checkpoint that normally runs first carries the
    distilled memory over -- but that checkpoint is skipped when checkpointing is
    disabled or the conversation was never seeded, and then only SharedState
    survives. Claude resumes by ``session_id`` and keeps everything.

    Args:
        allowed_intents: The emitting role's intent set; becomes the enforced
            ``intent_type`` enum. Required, because a backend that guesses it
            is exactly the drift this transport exists to prevent.
        model: Codex model id.
        cwd: Session-private working directory for the agent's own scratch
            files. Deliberately not the session directory: under the
            ``workspace-write`` sandbox the model may read the whole
            filesystem but write only its declared roots, so session state
            stays out of reach.
        writable_roots: Extra roots the session may write; ``cwd`` is always
            included.
        sandbox_mode: One of ``CODEX_SANDBOX_MODES``; blank defers to the
            deployment's ``HYPERLOOM_CODEX_SANDBOX_MODE``.
        codex_bin: Optional Codex runtime path; the SDK resolves its own when
            empty.
        call_timeout_s: Wall-clock cap for one ``run()``. Env override:
            ``INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC``.
        env: Values overlaid on ``os.environ`` for the child process.
        retry_policy: Bounded backoff for transient SDK/gateway failures; env
            override via ``INFERENCE_OPTIMIZER_LLM_RETRY_*`` (ATTEMPTS=1
            disables), shared with the Claude path.
    """

    allowed_intents: frozenset[IntentType]
    model: str = DEFAULT_CODEX_MODEL
    cwd: Path = field(default_factory=Path.cwd)
    writable_roots: tuple[Path, ...] = ()
    sandbox_mode: str = ""
    codex_bin: str = ""
    # An agent turn carries a tool loop, so it needs the conversational budget
    # the Claude orchestration path also floors at, not a completion's 120s.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC",
            default=300.0,
        )
    )
    env: dict[str, str] | None = None
    # Bounded transient-failure retry/backoff, on the same
    # INFERENCE_OPTIMIZER_LLM_RETRY_* knobs the Claude path reads.
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

    name: str = "codex"
    calls: list[dict[str, Any]] = field(default_factory=list)

    # Every turn runs on one held SDK thread, so the Coordinator's delta gating
    # and checkpoint compaction both apply. Not a dataclass field: a
    # session-scoped backend has no stateless mode to fall back to.
    conversational = True
    # Which prompt modules describe a surface this backend actually has. Read
    # by the prompt builder, which cannot infer it from the role: the
    # orchestration role is Claude on paper and Codex in an OpenAI-only run.
    transport = TRANSPORT_STRUCTURED_OUTPUT

    _session: CodexSession | None = field(default=None, init=False, repr=False)
    # Developer instructions the open thread was started with. The orchestration
    # system prompt is re-scoped on phase entry, and thread instructions are
    # fixed at thread_start, so a change has to open a new thread.
    _thread_instructions: str = field(default="", init=False, repr=False)
    # Whether the open thread has carried a turn. A fresh thread holds no
    # history, so the caller owes it a full push rather than a delta.
    _thread_seeded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize and secure the session-private runtime root.

        Raises:
            BackendError: If ``allowed_intents`` is empty or the working
                directory cannot be prepared.
        """
        if not self.allowed_intents:
            raise BackendError("CodexBackend requires the emitting role's allowed_intents")
        self.cwd = Path(self.cwd).expanduser().resolve()
        try:
            self.cwd.mkdir(parents=True, mode=0o700, exist_ok=True)
            self.cwd.chmod(0o700)
        except OSError as exc:
            raise BackendError(f"cannot prepare Codex backend cwd {self.cwd}: {exc}") from exc
        roots = tuple(Path(root).expanduser().resolve() for root in self.writable_roots)
        if self.cwd not in roots:
            roots = (self.cwd, *roots)
        self.writable_roots = roots

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
        allow_no_intent: bool = False,
    ) -> BackendTurnResult:
        """Run one Codex Agent SDK turn and parse its enforced intent envelope.

        Args:
            prompt: The composed turn prompt; sent as the user turn.
            system_prompt: The role's system prompt; sent as thread developer
                instructions alongside the transport contract.
            tools: Unused. The Codex session mounts the SDK's own tool surface;
                the Claude tool names PolicyGate returns have no counterpart
                here.
            max_turns: Unused. The SDK owns the agent loop within one turn.
            allow_no_intent: When ``True`` the turn is not schema-constrained
                and its text is returned unparsed. Checkpoint/compaction turns
                ask for a free-form summary, which the intent envelope cannot
                carry.

        Returns:
            BackendTurnResult: The validated intents plus raw reply text and
            model/usage metadata.

        Raises:
            BackendError: If the session's config or sandbox policy is
                unusable. Not retried: a misconfigured deployment does not
                become usable on the second attempt.
            LLMCallFailed: If the SDK turn kept failing once the retry budget
                was spent, or the turn reported an in-band error.
            NoIntentEmitted: If the reply did not honour the enforced schema or
                failed intent validation.
        """
        output_schema = None if allow_no_intent else build_intent_envelope_schema(self.allowed_intents)
        # Dropping the schema is not enough on its own: the thread's developer
        # instructions carry the OUTPUT FORMAT block for the life of the thread
        # and cannot be scoped out for one turn, so a checkpoint turn asking for
        # a different JSON shape would be answered with an intent envelope. The
        # compaction parser reads that as degenerate, skips the compaction, and
        # leaves the conversation growing -- the failure this backend exists to
        # end. An explicit per-turn override is the only channel available.
        turn_prompt = f"{_PER_TURN_SCHEMA_OVERRIDE}\n\n{prompt}" if allow_no_intent else prompt

        async def _one_attempt() -> Any:
            """Acquire the session and run one turn under the retry policy."""
            session = await self._session_for(system_prompt)
            return await session.turn(
                turn_prompt,
                timeout_sec=self.call_timeout_s,
                output_schema=output_schema,
            )

        def _note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            """Record a transient-failure retry warning into the call log."""
            self.calls.append(
                {"warn": f"codex SDK transient failure (attempt {attempt}): {exc!r}; retrying in {delay:.2f}s"}
            )

        try:
            sdk_result = await retry_with_backoff(
                _one_attempt,
                policy=self.retry_policy,
                retry_on=(CodexSessionError,),
                on_retry=_note_retry,
            )
        except CodexSessionError as exc:
            raise LLMCallFailed(f"Codex Agent SDK turn failed: {redact_secret_values(str(exc))}") from exc
        if sdk_result.error:
            raise LLMCallFailed("Codex Agent SDK turn failed: " + redact_secret_values(sdk_result.error))
        self._thread_seeded = True

        usage = dict(sdk_result.usage or {})
        input_tokens = safe_int(usage.get("input_tokens"))
        output_tokens = safe_int(usage.get("output_tokens"))
        cache_read_tokens = safe_int(usage.get("cache_read_input_tokens"))
        reasoning_tokens = safe_int(usage.get("reasoning_output_tokens"))
        self.calls.append(
            {
                "model": self.model,
                "prompt_chars": len(prompt),
                "reply_chars": len(sdk_result.text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "thread_id": sdk_result.thread_id,
            }
        )
        metadata: dict[str, Any] = {
            "model": self.model,
            "thread_id": sdk_result.thread_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            # Codex reports per-turn counts, so the input side already is this
            # request's context size — the figure the checkpoint policy compares
            # against the model's window. The cached tokens are part of that
            # count, not an addition to it: Codex reports 5152 input / 3072
            # cached / 16 output as 5168 total. Adding them back inflated the
            # water level by the cache hit rate, which on a long conversation is
            # most of the prompt, so the policy compacted early — and compaction
            # resets the conversation this backend exists to keep.
            "context_tokens_peak": input_tokens,
            # Stated by Codex per turn, and better than any table this side
            # keeps: the compaction trigger is a fraction of it. Written
            # unconditionally — 0 means "not reported", which is what
            # adopt_context_window ignores.
            "model_context_window": safe_int(usage.get("model_context_window")),
            # Full conversation text for conversations.jsonl, handed up for
            # the caller to persist.
            "prompt": prompt,
            "response": sdk_result.text,
        }

        if allow_no_intent:
            return BackendTurnResult(intents=[], raw_text=sdk_result.text, metadata=metadata)
        envelope = decode_intent_envelope(sdk_result.text)
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"codex envelope invalid: {exc}") from exc
        return BackendTurnResult(intents=intents, raw_text=sdk_result.text, metadata=metadata)

    # ------------------------------------------------------------------
    @property
    def needs_seed(self) -> bool:
        """True when the open conversation has no history to build a delta on.

        The caller decides between a full push and a delta, but only this
        backend knows when the thread underneath was replaced — by a reset, by
        a re-scoped system prompt, or by a turn that never landed.

        This is the question a backend can answer without knowing what the turn
        will carry. The Coordinator reads it only from backends that cannot
        answer :meth:`needs_seed_for`, so this backend's own answer is never the
        one used; it stays because the attribute is the fallback half of that
        protocol, and a backend implementing only this half must still work.
        """
        return not self._thread_seeded

    def needs_seed_for(self, system_prompt: str | None) -> bool:
        """True when the turn about to run will start on an empty thread.

        :attr:`needs_seed` can only describe the thread as it stands, but the
        caller has to choose SEED or DELTA *before* the turn, and a re-scoped
        system prompt replaces the thread inside the turn itself. Answering for
        the prompt that is about to be sent is what keeps a thin delta out of a
        thread that holds none of the state the delta omits.
        """
        if not self._thread_seeded:
            return True
        return self._instructions_for(system_prompt) != self._thread_instructions

    def _instructions_for(self, system_prompt: str | None) -> str:
        """Thread-level instructions implied by one system prompt."""
        return "\n\n".join(
            part for part in ((system_prompt or "").strip(), build_output_instructions(self.allowed_intents)) if part
        )

    def reset_conversation(self) -> None:
        """Start the next turn on a fresh conversation.

        The conversation is the SDK thread; the client and its private state
        directory stay up, so a compaction cannot leak a child process.
        """
        self._thread_seeded = False
        if self._session is not None:
            self._session.reset_thread()

    async def aclose(self) -> None:
        """Release the held session: SDK client, child process, ``CODEX_HOME``."""
        session, self._session = self._session, None
        self._thread_seeded = False
        if session is not None:
            await session.aclose()

    async def _session_for(self, system_prompt: str | None) -> CodexSession:
        """Return the held session, opening it (or re-scoping it) as needed.

        Raises:
            BackendError: If the session's config or sandbox policy is unusable,
                which is the same on every attempt.
            CodexSessionError: If the SDK client could not be opened. Raised as
                it is, so the caller's retry wrapper treats a refused connection
                the way it treats one that drops mid-turn.
        """
        instructions = self._instructions_for(system_prompt)
        if self._session is None:
            self._session = CodexSession(
                cwd=self.cwd,
                model=self.model,
                developer_instructions=instructions,
                writable_roots=self.writable_roots,
                sandbox_mode=self.sandbox_mode,
                codex_bin=self.codex_bin,
                env=self.env,
            )
            self._thread_seeded = False
            try:
                await self._session.start()
            except CodexSessionUnavailableError as exc:
                # A config or sandbox fault, identical on every attempt.
                self._session = None
                raise BackendError(redact_secret_values(str(exc))) from exc
            except CodexSessionError:
                # Left as it is so the retry wrapper around the attempt can see
                # it. Opening the client is the first leg of the turn and fails
                # for the same transient reasons the turn does; converting it
                # here meant a gateway that refused one connection ended the
                # turn, while the identical error from turn() was retried.
                self._session = None
                raise
        elif instructions != self._thread_instructions:
            self._session.developer_instructions = instructions
            self._session.reset_thread()
            self._thread_seeded = False
        self._thread_instructions = instructions
        return self._session


__all__ = ["CodexBackend", "build_output_instructions", "decode_intent_envelope"]
