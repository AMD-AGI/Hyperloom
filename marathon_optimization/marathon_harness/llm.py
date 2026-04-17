"""LLM client — Claude Code SDK, Primus-Claw sessions, or Claude CLI fallback.

Three backends in priority order:
  1. Primus-Claw (claw_url set) — sends prompts via Claw session API, executor
     runs Claude Code SDK internally.  No Anthropic API key needed on this side.
  2. claude_code_sdk — direct agentic tool-use when the SDK + API key are present.
  3. claude CLI — same binary as interactive Claude Code, subprocess fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup
else:
    try:
        from exceptiongroup import BaseExceptionGroup as _BaseExceptionGroup
    except ImportError:
        _BaseExceptionGroup = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

CLAUDE_CLI_TIMEOUT_S = 3600
CLAW_SSE_TIMEOUT_S = 900  # 15 min per Claw call; OOB round timeout handles overall cap

_HAS_SDK = False
try:
    import claude_code_sdk  # noqa: F401
    _HAS_SDK = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Monkey-patch: make the SDK skip unknown message types instead of crashing.
# ---------------------------------------------------------------------------
_sdk_patched = False

def _patch_sdk_message_parser() -> None:
    global _sdk_patched
    if _sdk_patched or not _HAS_SDK:
        return
    try:
        import claude_code_sdk._internal.client as _client_mod
        import claude_code_sdk._internal.message_parser as _parser_mod
        _original_parse = _parser_mod.parse_message

        def _tolerant_parse(data: dict) -> Any:
            try:
                return _original_parse(data)
            except (KeyError, ValueError, TypeError):
                log.debug("Skipping unrecognised SDK message: type=%s", data.get("type"))
                return None
            except Exception as _exc:
                if "unknown" in str(_exc).lower() or "message type" in str(_exc).lower():
                    log.debug("Skipping unrecognised SDK message: type=%s", data.get("type"))
                    return None
                raise

        _parser_mod.parse_message = _tolerant_parse
        _client_mod.parse_message = _tolerant_parse
        _sdk_patched = True
        log.debug("Patched claude_code_sdk message parser for unknown-type tolerance")
    except Exception as exc:
        log.warning("Could not patch SDK message parser: %s", exc)


@dataclass
class LLMResult:
    text: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Primus-Claw session client
# ---------------------------------------------------------------------------

class ClawClient:
    """Thin client for the Primus-Claw Claw REST + SSE API.

    Flow per call:
      1. POST /v1/sessions               → create session
      2. POST /v1/sessions/{id}/messages  → send prompt (triggers executor)
      3. GET  /v1/chat/sessions/{id}/messages  → SSE stream until agentStatus=stopped
      4. DELETE /v1/sessions/{id}         → cleanup
    """

    def __init__(self, base_url: str, auth_token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._headers: dict[str, str] = {}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        timeout_s: float = CLAW_SSE_TIMEOUT_S,
    ) -> LLMResult:
        """Create a session, send the prompt, stream events, return the result."""
        import httpx

        t0 = time.monotonic()
        session_id = ""

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(30.0, read=timeout_s),
                verify=False,
            ) as client:
                # 1) Create session
                create_body: dict[str, Any] = {
                    "name": prompt[:60],
                    "mode": "claw",
                }
                if system_prompt:
                    create_body["system_prompt"] = system_prompt
                resp = await client.post("/v1/sessions", json=create_body)
                resp.raise_for_status()
                session_data = resp.json().get("data", resp.json())
                session_id = session_data["session_id"]
                log.info("Claw session created: %s", session_id)

                try:
                    # 2) Send message (triggers executor)
                    msg_body = {
                        "content": prompt,
                        "messageType": "text",
                        "taskMode": "agent",
                    }
                    resp = await client.post(
                        f"/v1/sessions/{session_id}/messages",
                        json=msg_body,
                    )
                    resp.raise_for_status()
                    log.info("Claw message sent to session %s", session_id)

                    # 3) Subscribe to SSE stream
                    result = await self._consume_sse(client, session_id, timeout_s)
                    result.duration_ms = int((time.monotonic() - t0) * 1000)
                    return result

                finally:
                    # Always clean up session, even on exception/cancellation
                    try:
                        await asyncio.wait_for(
                            client.delete(f"/v1/sessions/{session_id}"),
                            timeout=10,
                        )
                    except Exception:
                        log.debug("Session cleanup failed for %s", session_id)

        except BaseException as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            # httpx 0.28+ wraps CancelledError inside ExceptionGroup.
            # Unwrap and re-raise so shutdown signals propagate correctly.
            if isinstance(exc, asyncio.CancelledError):
                log.warning("Claw call cancelled (session=%s, %dms)", session_id or "(none)", duration_ms)
                raise
            if _BaseExceptionGroup is not None and isinstance(exc, _BaseExceptionGroup):
                cancelled = exc.subgroup(asyncio.CancelledError)
                if cancelled is not None:
                    log.warning("Claw call cancelled [wrapped] (session=%s, %dms)", session_id or "(none)", duration_ms)
                    raise asyncio.CancelledError() from cancelled
            if not isinstance(exc, Exception):
                raise
            log.error("Claw call failed (session=%s): %s", session_id or "(none)", exc)
            return LLMResult(
                is_error=True,
                error_message=f"Claw error: {exc}",
                duration_ms=duration_ms,
            )

    async def _consume_sse(
        self,
        client: Any,
        session_id: str,
        timeout_s: float,
    ) -> LLMResult:
        """Read the SSE stream until the agent stops or we time out."""
        import httpx

        text_chunks: list[str] = []
        final_text = ""
        tool_events: list[dict[str, Any]] = []
        got_stopped = False
        turns = 0

        deadline = time.monotonic() + timeout_s
        last_event_time = time.monotonic()
        # If no SSE event arrives within this window, assume the session is
        # dead/garbage-collected and bail out instead of blocking until the
        # global httpx read timeout (which can be 3600s).
        idle_timeout_s = 120
        url = f"/v1/chat/sessions/{session_id}/messages"

        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            event_name = ""
            data_buf = ""

            async for raw_line in resp.aiter_lines():
                last_event_time = time.monotonic()
                if time.monotonic() > deadline:
                    log.warning("Claw SSE timeout after %.0fs", timeout_s)
                    break

                line = raw_line.rstrip("\r\n")

                if line.startswith(": "):
                    continue

                if line.startswith("event: "):
                    event_name = line[7:]
                    continue

                if line.startswith("data: "):
                    data_buf = line[6:]
                    continue

                if line == "" and data_buf:
                    try:
                        payload = json.loads(data_buf)
                    except json.JSONDecodeError:
                        data_buf = ""
                        event_name = ""
                        continue

                    self._handle_event(
                        event_name or payload.get("type", ""),
                        payload,
                        text_chunks,
                        tool_events,
                    )

                    evt_type = payload.get("type", "")
                    if evt_type == "statusUpdate" and payload.get("agentStatus") == "stopped":
                        got_stopped = True
                    if evt_type == "chat" and payload.get("sender") == "assistant":
                        turns += 1
                        final_text = self._extract_text(payload)
                    if evt_type == "exec_complete":
                        got_stopped = True

                    data_buf = ""
                    event_name = ""

                    if got_stopped:
                        break

        assembled = final_text or "".join(text_chunks).strip()
        if not assembled and not got_stopped:
            return LLMResult(
                is_error=True,
                error_message="Claw session ended with no output",
            )

        return LLMResult(
            text=assembled,
            num_turns=max(turns, 1),
        )

    def _handle_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        text_chunks: list[str],
        tool_events: list[dict[str, Any]],
    ) -> None:
        evt_type = payload.get("type", event_name)

        if evt_type in ("chatDelta", "chat_delta"):
            delta = payload.get("delta", {})
            chunk = delta.get("content", "")
            if chunk:
                text_chunks.append(chunk)

        elif evt_type in ("toolUsed", "tool_used"):
            status = payload.get("status", "")
            tool = payload.get("tool", "")
            if status in ("start", "success", "error"):
                log.info("Claw tool: %s [%s] %s",
                         tool, status, (payload.get("brief") or "")[:80])
            tool_events.append(payload)
            if len(tool_events) > 200:
                tool_events[:] = tool_events[-100:]

        elif evt_type == "error":
            log.error("Claw error event: %s", payload.get("message", ""))

        elif evt_type in ("liveStatus", "live_status"):
            log.debug("Claw status: %s", payload.get("text", ""))

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        """Extract assistant text from a 'chat' event."""
        tp = payload.get("text_preview", "")
        content = payload.get("content", "")
        if isinstance(content, dict):
            blocks = content.get("content", [])
            if isinstance(blocks, list):
                parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
                return "\n".join(parts)
        if isinstance(content, str) and content:
            return content
        return tp


# ---------------------------------------------------------------------------
# Main LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """LLM client: Primus-Claw → claude_code_sdk → claude CLI (priority order)."""

    MAX_RETRIES = 3
    BACKOFF_BASE_S = 2.0

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        cwd: str = "/sgl-workspace",
        add_dirs: list[str] | None = None,
        env: dict[str, str] | None = None,
        system_prompt: str = "",
        inferencex_path: str = "",
        base_dir: str = "",
        claw_url: str = "",
    ):
        self.model = model
        self.cwd = cwd
        self.add_dirs = add_dirs or ["/shared_nfs/nehaprakriya"]
        self.env = env or {}
        self.default_system_prompt = system_prompt
        self.inferencex_path = inferencex_path
        self.base_dir = base_dir
        self._total_cost = 0.0
        self._total_calls = 0
        self._total_turns = 0

        # Claw backend
        self._claw: ClawClient | None = None
        if claw_url:
            auth = self.env.get("CLAW_AUTH_TOKEN", "")
            self._claw = ClawClient(claw_url, auth_token=auth)

        # CLI fallback
        _bin = (self.env.get("CLAUDE_CODE_BIN") or "").strip()
        if _bin and Path(_bin).is_file():
            self._claude_cli = _bin
        else:
            self._claude_cli = shutil.which(_bin or "claude")

        if not self._claw and not _HAS_SDK and not self._claude_cli:
            log.error(
                "No LLM driver: set --claw-url for Primus-Claw, "
                "install claude_code_sdk (pip install claude-code-sdk), "
                "and/or install Claude Code CLI (npm install -g @anthropic-ai/claude-code)."
            )
        _patch_sdk_message_parser()

        if self._claw:
            backend = f"primus-claw ({claw_url})"
        elif _HAS_SDK:
            backend = "claude_code_sdk"
        elif self._claude_cli:
            backend = "claude_cli"
        else:
            backend = "none"
        log.info("LLM backend: %s (model=%s)", backend, self.model)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_turns(self) -> int:
        return self._total_turns

    async def call(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_turns: int = 30,
        output_file: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> LLMResult:
        """Execute a scoped LLM call.  Returns parsed LLMResult."""
        if output_file:
            prompt = prompt.replace("$OUTPUT_FILE", output_file)
            prompt = prompt.replace("$RESULT_DIR", str(Path(output_file).parent))
        if self.inferencex_path:
            prompt = prompt.replace("$INFERENCEX_PATH", self.inferencex_path)
        if self.base_dir:
            prompt = prompt.replace("$BASE_DIR", self.base_dir)

        if self._claw:
            return await self._call_claw(
                prompt, system_prompt=system_prompt, output_file=output_file,
            )

        if _HAS_SDK:
            return await self._call_sdk(
                prompt, system_prompt=system_prompt, max_turns=max_turns,
                output_file=output_file, model=model, cwd=cwd, allowed_tools=allowed_tools,
            )

        return await self._call_claude_cli(
            prompt, system_prompt=system_prompt, output_file=output_file,
            model=model, cwd=cwd,
        )

    # ------------------------------------------------------------------
    # Primus-Claw backend
    # ------------------------------------------------------------------

    async def _call_claw(
        self,
        prompt: str,
        *,
        system_prompt: str,
        output_file: str | None,
    ) -> LLMResult:
        assert self._claw is not None
        effective_system = system_prompt or self.default_system_prompt or ""

        full_prompt = prompt
        if effective_system:
            full_prompt = f"{effective_system}\n\n---\n\n{prompt}"
        if output_file:
            full_prompt += (
                f"\n\nWhen done, write the final JSON object to this path exactly: {output_file}"
            )

        last_err = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            log.info("LLM/Claw attempt %d/%d", attempt, self.MAX_RETRIES)
            result = await self._claw.run(full_prompt)

            if not result.is_error:
                self._total_calls += 1
                self._total_turns += result.num_turns
                log.info("LLM/Claw done: %d turns, %dms, %d chars",
                         result.num_turns, result.duration_ms, len(result.text))

                if output_file:
                    result.output = self._read_output_file(output_file)
                    if not result.output:
                        result.output = self._parse_json_from_text(result.text)
                return result

            last_err = result.error_message
            log.warning("LLM/Claw attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, last_err)
            if attempt < self.MAX_RETRIES:
                await asyncio.sleep(self.BACKOFF_BASE_S ** attempt)

        return LLMResult(
            is_error=True,
            error_message=f"All {self.MAX_RETRIES} Claw attempts failed: {last_err}",
        )

    # ------------------------------------------------------------------
    # SDK backend (agentic, tool-use)
    # ------------------------------------------------------------------

    async def _call_sdk(
        self, prompt: str, *, system_prompt: str, max_turns: int,
        output_file: str | None, model: str | None, cwd: str | None,
        allowed_tools: list[str] | None,
    ) -> LLMResult:
        from claude_code_sdk import query, ClaudeCodeOptions

        effective_system_prompt = system_prompt or self.default_system_prompt or None
        opts = ClaudeCodeOptions(
            model=model or self.model,
            max_turns=max_turns,
            cwd=cwd or self.cwd,
            system_prompt=effective_system_prompt,
            permission_mode="acceptEdits",
        )
        if allowed_tools:
            opts.allowed_tools = allowed_tools

        last_err = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            t0 = time.monotonic()
            log.info("LLM/SDK call attempt %d/%d (max_turns=%d)", attempt, self.MAX_RETRIES, max_turns)
            try:
                messages: list[Any] = []
                async for msg in query(prompt=prompt, options=opts):
                    if msg is not None:
                        if not messages:
                            log.info("LLM stream: first message received (type=%s)",
                                     type(msg).__name__)
                        messages.append(msg)

                if not messages:
                    raise RuntimeError("No messages received from SDK")

                duration_ms = int((time.monotonic() - t0) * 1000)
                text_parts: list[str] = []
                for m in messages:
                    if hasattr(m, "content") and isinstance(m.content, str):
                        text_parts.append(m.content)
                full_text = "\n".join(text_parts)

                cost = self._extract_cost(messages)
                turns = len([m for m in messages if getattr(m, "role", "") == "assistant"])
                self._total_cost += cost
                self._total_calls += 1
                self._total_turns += turns

                log.info("LLM/SDK done: %d msgs, %d turns, %dms, $%.4f",
                         len(messages), turns, duration_ms, cost)

                output = {}
                if output_file:
                    output = self._read_output_file(output_file)

                return LLMResult(text=full_text, output=output, cost_usd=cost,
                                 num_turns=turns, duration_ms=duration_ms)
            except Exception as exc:
                last_err = str(exc)
                log.warning("LLM/SDK attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, last_err)
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.BACKOFF_BASE_S ** attempt)

        return LLMResult(is_error=True, error_message=f"All {self.MAX_RETRIES} SDK attempts failed: {last_err}")

    # ------------------------------------------------------------------
    # Claude CLI (no SDK) — same binary as interactive Claude Code
    # ------------------------------------------------------------------

    async def _call_claude_cli(
        self,
        prompt: str,
        *,
        system_prompt: str,
        output_file: str | None,
        model: str | None,
        cwd: str | None,
    ) -> LLMResult:
        if not self._claude_cli:
            return LLMResult(
                is_error=True,
                error_message="claude_code_sdk not installed and `claude` CLI not found (see logs).",
            )

        effective_system = system_prompt or self.default_system_prompt or ""
        full_prompt = prompt
        if effective_system:
            full_prompt = f"{effective_system}\n\n---\n\n{prompt}"
        if output_file:
            full_prompt += (
                f"\n\nWhen done, write the final JSON object to this path exactly: {output_file}"
            )

        use_model = model or self.model
        work_cwd = cwd or self.cwd
        merged = {**os.environ, **self.env}

        last_err = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            t0 = time.monotonic()
            cmd: list[str] = [
                self._claude_cli,
                "-p",
                full_prompt,
                "--permission-mode",
                "acceptEdits",
                "--output-format",
                "text",
            ]
            if use_model:
                cmd.extend(["--model", use_model])
            for d in {work_cwd, *self.add_dirs}:
                p = Path(d)
                if p.is_dir():
                    cmd.extend(["--add-dir", str(p)])

            log.info("LLM/CLI attempt %d/%d (cwd=%s)", attempt, self.MAX_RETRIES, work_cwd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=work_cwd,
                    env=merged,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=CLAUDE_CLI_TIMEOUT_S,
                )
                duration_ms = int((time.monotonic() - t0) * 1000)
                text = (stdout_b or b"").decode("utf-8", errors="replace")
                err_txt = (stderr_b or b"").decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    raise RuntimeError(f"claude exited {proc.returncode}: {err_txt[:2000]}")

                output: dict[str, Any] = {}
                if output_file and Path(output_file).exists():
                    output = self._read_output_file(output_file)
                if not output:
                    output = self._parse_json_from_text(text)

                self._total_calls += 1
                self._total_turns += 1
                log.info("LLM/CLI done: %d chars stdout, %dms", len(text), duration_ms)
                return LLMResult(
                    text=text, output=output, cost_usd=0.0, num_turns=1, duration_ms=duration_ms,
                )
            except Exception as exc:
                last_err = str(exc)
                log.warning("LLM/CLI attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, last_err)
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.BACKOFF_BASE_S ** attempt)

        return LLMResult(
            is_error=True,
            error_message=f"All {self.MAX_RETRIES} CLI attempts failed: {last_err}",
        )

    @staticmethod
    def _parse_json_from_text(text: str) -> dict[str, Any]:
        """Extract a JSON object from LLM response text."""
        stripped = text.strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        import re
        for pattern in [r'```json\s*\n(.*?)\n\s*```', r'```\s*\n(.*?)\n\s*```']:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue

        brace_start = text.find('{')
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[brace_start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            break
                        break

        log.warning("Could not extract JSON from LLM response (%d chars)", len(text))
        return {}

    def sync_stats(self, state: Any) -> None:
        """Push accumulated cost/calls/turns into MarathonState."""
        state.total_llm_cost_usd = self._total_cost
        state.total_llm_calls = self._total_calls
        state.total_llm_turns = self._total_turns

    @staticmethod
    def _extract_cost(messages: list[Any]) -> float:
        for m in reversed(messages):
            if hasattr(m, "cost_usd"):
                return float(m.cost_usd)
            if hasattr(m, "total_cost_usd") and m.total_cost_usd:
                return float(m.total_cost_usd)
            if hasattr(m, "usage") and isinstance(m.usage, dict):
                inp = m.usage.get("input_tokens", 0)
                out = m.usage.get("output_tokens", 0)
                return (inp * 3 + out * 15) / 1_000_000
        return 0.0

    @staticmethod
    def _read_output_file(path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            log.warning("LLM output file not written: %s", path)
            return {}
        try:
            data = json.loads(p.read_text())
            if not isinstance(data, dict):
                log.warning("LLM output file is not a JSON object: %s", path)
                return {}
            return data
        except json.JSONDecodeError as exc:
            log.warning("LLM output file has invalid JSON (%s): %s", exc, path)
            return {}
        except OSError as exc:
            log.warning("Cannot read LLM output file (%s): %s", exc, path)
            return {}
