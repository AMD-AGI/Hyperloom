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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.codex_session import (
    CodexSessionError,
    resolve_codex_sandbox_mode,
    run_codex_turn,
)
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .agent_role import DEFAULT_CODEX_MODEL
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    parse_call_timeout_env,
    safe_int,
)
from .mcp_emit_intent import build_intent_envelope_schema, payload_contract


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
    heartbeat = json.dumps({"topic": "heartbeat", "body_md": "ok"})
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
Your final message MUST be exactly one JSON object matching the enforced
output schema — no prose, no code fences, nothing around it:

{{"intents": [{{"intent_type": "...", "payload": "..."}}]}}

- `payload` is a STRING holding a serialized JSON object. The schema cannot
  express a free-form object, so serialize the payload and escape it.
- Required keys per intent_type: {contract}.
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

    name: str = "codex"
    calls: list[dict[str, Any]] = field(default_factory=list)

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

        Returns:
            BackendTurnResult: The validated intents plus raw reply text and
            model/usage metadata.

        Raises:
            BackendError: If the sandbox policy is unusable.
            LLMCallFailed: If the SDK turn failed or reported an in-band error.
            NoIntentEmitted: If the reply did not honour the enforced schema or
                failed intent validation.
        """
        try:
            resolved_sandbox = resolve_codex_sandbox_mode(sandbox_mode=self.sandbox_mode, env=self.env)
        except CodexSessionError as exc:
            raise BackendError(redact_secret_values(str(exc))) from exc

        output_schema = build_intent_envelope_schema(self.allowed_intents)
        developer_instructions = "\n\n".join(
            part for part in ((system_prompt or "").strip(), build_output_instructions(self.allowed_intents)) if part
        )
        try:
            sdk_result = await run_codex_turn(
                prompt=prompt,
                developer_instructions=developer_instructions,
                cwd=self.cwd,
                model=self.model,
                timeout_sec=self.call_timeout_s,
                writable_roots=self.writable_roots,
                sandbox_mode=resolved_sandbox,
                codex_bin=self.codex_bin,
                output_schema=output_schema,
                env=self.env,
            )
        except CodexSessionError as exc:
            raise LLMCallFailed(f"Codex Agent SDK turn failed: {redact_secret_values(str(exc))}") from exc
        if sdk_result.error:
            raise LLMCallFailed("Codex Agent SDK turn failed: " + redact_secret_values(sdk_result.error))

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

        envelope = decode_intent_envelope(sdk_result.text)
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"codex envelope invalid: {exc}") from exc

        metadata: dict[str, Any] = {
            "model": self.model,
            "thread_id": sdk_result.thread_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            # Full conversation text for conversations.jsonl, handed up for
            # the caller to persist.
            "prompt": prompt,
            "response": sdk_result.text,
        }
        return BackendTurnResult(intents=intents, raw_text=sdk_result.text, metadata=metadata)


__all__ = ["CodexBackend", "build_output_instructions", "decode_intent_envelope"]
