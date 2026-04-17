"""OOB backend dispatch — GEAK (MCP), Codex (agent), Claude (agent), Claw (session).

Each dispatcher returns an OOBResult.  Backend rotation + failure tracking included.

When *claw_url* is set the codex/claude/geak dispatchers route through Primus-Claw
sessions instead of calling claude_code_sdk directly — no Anthropic API key needed
on the marathon side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEAK_POLL_INTERVAL_S = 30
GEAK_POLL_TIMEOUT_MIN = 30
GEAK_STEP_LIMIT = 100

CODEX_TIMEOUT_MIN = 10
CLAUDE_TIMEOUT_MIN = 15

OOB_POLL_INTERVAL_S = 15

BACKEND_FAILURE_THRESHOLD = 5

CLASSIFICATION: dict[str, list[str]] = {
    "dispatch-fix":                      ["local"],
    "config-only":                       ["local"],
    "oob-rewrite":                       ["geak", "codex", "claude"],
    "oob-rewrite-register-constrained":  ["codex", "claude"],
    "triton-rewrite":                    ["geak", "codex", "claude"],
    "hip-kernel":                        ["geak", "claude"],
    "framework-scheduling":              ["claude"],
    "kernel-fusion":                     ["geak", "claude"],
}

BROKEN_MODELS = {"gpt-5.2"}


@dataclass
class OOBResult:
    backend: str = ""
    status: str = ""      # success | compile_fail | timeout | error
    code: str | None = None
    error: str | None = None
    duration_s: float = 0.0
    model: str | None = None


class OOBBackends:
    """Manages dispatch to OOB backends with failure tracking.

    When *claw_url* is provided all agent dispatchers (geak, codex, claude)
    route through Primus-Claw sessions.  Otherwise they use claude_code_sdk
    directly (requires ANTHROPIC_API_KEY).
    """

    def __init__(self, env: dict[str, str] | None = None, claw_url: str = ""):
        self.env = env or dict(os.environ)
        self.failure_counts: dict[str, int] = {
            "geak": 0, "codex": 0, "claude": 0,
        }
        self.claw_url = claw_url

    def select_backends(
        self,
        strategy: str,
        round_num: int,
        session_history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        eligible = CLASSIFICATION.get(strategy, ["claude"])
        eligible = [b for b in eligible if b != "local"]

        eligible = [b for b in eligible
                    if self.failure_counts.get(b, 0) < BACKEND_FAILURE_THRESHOLD]

        if not eligible:
            eligible = ["claude"]

        if round_num == 1:
            return eligible
        elif round_num <= 3:
            best = self._best_backend(session_history or [])
            others = [b for b in eligible if b != best]
            return ([best] if best in eligible else []) + others[:1]
        elif round_num == 4:
            return eligible[:2]
        else:
            best = self._best_backend(session_history or [])
            return [best] if best in eligible else eligible[:1]

    async def dispatch(
        self,
        backend: str,
        prompt: str,
        target: dict[str, Any],
        files: dict[str, str] | None = None,
    ) -> OOBResult:
        try:
            if backend == "geak":
                return await self._dispatch_geak(prompt, target, files)
            elif backend == "codex":
                return await self._dispatch_codex(prompt, target)
            elif backend == "claude":
                result = await self._dispatch_claude(prompt, target)
                if result.status == "error":
                    log.warning("Claude failed, falling back to Codex")
                    return await self._dispatch_codex(prompt, target)
                return result
            else:
                return OOBResult(backend=backend, status="error", error=f"Unknown backend: {backend}")
        except asyncio.CancelledError:
            log.warning("OOB dispatch cancelled (backend=%s)", backend)
            return OOBResult(backend=backend, status="error", error="cancelled (shutdown)")
        except Exception as exc:
            self.failure_counts[backend] = self.failure_counts.get(backend, 0) + 1
            return OOBResult(backend=backend, status="error", error=str(exc))

    def record_result(self, backend: str, success: bool) -> None:
        if success:
            self.failure_counts[backend] = 0
        else:
            self.failure_counts[backend] = self.failure_counts.get(backend, 0) + 1

    # ------------------------------------------------------------------
    # Claw session helper (shared by all dispatchers when claw_url is set)
    # ------------------------------------------------------------------

    async def _dispatch_via_claw(self, prompt: str, backend_label: str) -> OOBResult:
        """Send prompt to Primus-Claw session, wait for result, extract code."""
        from .llm import ClawClient, _BASH_BLOCKLIST_PREAMBLE

        t0 = time.monotonic()
        auth = self.env.get("CLAW_AUTH_TOKEN", "")
        ca_bundle = self.env.get("CLAW_CA_BUNDLE", "")
        client = ClawClient(self.claw_url, auth_token=auth, ca_bundle=ca_bundle)
        prompt = _BASH_BLOCKLIST_PREAMBLE + "\n" + prompt
        result = await client.run(prompt)
        duration = time.monotonic() - t0

        if result.is_error:
            return OOBResult(
                backend=backend_label, status="error",
                error=result.error_message, duration_s=duration,
            )

        code = self._extract_code_block(result.text)
        if code:
            self.failure_counts[backend_label] = 0
            return OOBResult(backend=backend_label, status="success", code=code, duration_s=duration)

        return OOBResult(
            backend=backend_label, status="error",
            error="No code block in Claw response", duration_s=duration,
        )

    # ------------------------------------------------------------------
    # GEAK — 7 MCP tools (via SDK or Claw)
    # ------------------------------------------------------------------

    async def _dispatch_geak(
        self, prompt: str, target: dict[str, Any], files: dict[str, str] | None = None,
    ) -> OOBResult:
        geak_prompt = self._build_geak_meta_prompt(prompt, target, files)

        if self.claw_url:
            return await self._dispatch_via_claw(geak_prompt, "geak")

        return await self._dispatch_geak_sdk(geak_prompt)

    async def _dispatch_geak_sdk(self, geak_prompt: str) -> OOBResult:
        t0 = time.monotonic()
        try:
            from claude_code_sdk import query, ClaudeCodeOptions

            opts = ClaudeCodeOptions(
                max_turns=10,
                cwd="/tmp",
                permission_mode="acceptEdits",
            )
            opts.allowed_tools = [
                "mcp__geak__geak_create_task",
                "mcp__geak__geak_submit_task",
                "mcp__geak__geak_get_task",
                "mcp__geak__geak_get_outputs",
                "mcp__geak__geak_download_file",
                "mcp__geak__geak_list_tasks",
                "mcp__geak__geak_set_model_config",
            ]

            messages: list[Any] = []
            async for msg in query(prompt=geak_prompt, options=opts):
                messages.append(msg)

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["geak"] = 0
                return OOBResult(backend="geak", status="success", code=code, duration_s=duration)
            return OOBResult(backend="geak", status="error", error="No code in GEAK output", duration_s=duration)
        except Exception as exc:
            return OOBResult(backend="geak", status="error", error=str(exc), duration_s=time.monotonic() - t0)

    def _build_geak_meta_prompt(
        self, prompt: str, target: dict[str, Any], files: dict[str, str] | None,
    ) -> str:
        workspace = self.env.get("GEAK_WORKSPACE", "control-plane-moe")
        file_list = []
        for name, content in (files or {}).items():
            file_list.append({"filename": name, "content": content})

        return (
            f"Use the GEAK MCP tools to create and run a kernel optimization task.\n\n"
            f"1. Call geak_create_task with:\n"
            f"   - input_type: 'file'\n"
            f"   - prompt: the optimization instructions below\n"
            f"   - step_limit: {GEAK_STEP_LIMIT}\n"
            f"   - workspace_id: '{workspace}'\n"
            f"   - gpu_count: 1\n"
            f"   - files: {json.dumps(file_list)}\n"
            f"2. Call geak_submit_task\n"
            f"3. Poll with geak_get_task every {GEAK_POLL_INTERVAL_S}s "
            f"(timeout {GEAK_POLL_TIMEOUT_MIN}min)\n"
            f"4. On completion, call geak_get_outputs and return the optimized kernel code\n\n"
            f"--- OPTIMIZATION INSTRUCTIONS ---\n{prompt}"
        )

    # ------------------------------------------------------------------
    # Codex — agent_* MCP tools (via SDK or Claw)
    # ------------------------------------------------------------------

    async def _dispatch_codex(self, prompt: str, target: dict[str, Any]) -> OOBResult:
        meta_prompt = (
            f"Use the agent MCP tools to dispatch a Codex task.\n\n"
            f"1. Call agent_create_task with agent='codex' and the prompt below\n"
            f"2. Call agent_submit_task\n"
            f"3. Poll with agent_get_task every {OOB_POLL_INTERVAL_S}s "
            f"(timeout {CODEX_TIMEOUT_MIN}min)\n"
            f"4. On completion, call agent_get_outputs and return the optimized code\n\n"
            f"--- PROMPT ---\n{prompt}"
        )

        if self.claw_url:
            return await self._dispatch_via_claw(meta_prompt, "codex")

        return await self._dispatch_codex_sdk(meta_prompt)

    async def _dispatch_codex_sdk(self, meta_prompt: str) -> OOBResult:
        t0 = time.monotonic()
        try:
            from claude_code_sdk import query, ClaudeCodeOptions

            opts = ClaudeCodeOptions(max_turns=8, cwd="/tmp", permission_mode="acceptEdits")
            opts.allowed_tools = [
                "mcp__agent__agent_create_task",
                "mcp__agent__agent_submit_task",
                "mcp__agent__agent_get_task",
                "mcp__agent__agent_get_outputs",
                "mcp__agent__agent_download_file",
                "mcp__agent__agent_cancel_task",
            ]

            messages: list[Any] = []
            async for msg in query(prompt=meta_prompt, options=opts):
                messages.append(msg)

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["codex"] = 0
                return OOBResult(backend="codex", status="success", code=code, duration_s=duration)
            return OOBResult(backend="codex", status="error", error="No code", duration_s=duration)
        except Exception as exc:
            return OOBResult(backend="codex", status="error", error=str(exc), duration_s=time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Claude — agent_* MCP tools (via SDK or Claw, fallback → Codex)
    # ------------------------------------------------------------------

    async def _dispatch_claude(self, prompt: str, target: dict[str, Any]) -> OOBResult:
        meta_prompt = (
            f"Use the agent MCP tools to dispatch a Claude task.\n\n"
            f"1. Call agent_create_task with agent='claude', max_turns=30, "
            f"and the prompt below\n"
            f"2. Call agent_submit_task\n"
            f"3. Poll with agent_get_task every {OOB_POLL_INTERVAL_S}s "
            f"(timeout {CLAUDE_TIMEOUT_MIN}min)\n"
            f"4. On completion, call agent_get_outputs and return the optimized code\n\n"
            f"--- PROMPT ---\n{prompt}"
        )

        if self.claw_url:
            return await self._dispatch_via_claw(meta_prompt, "claude")

        return await self._dispatch_claude_sdk(meta_prompt)

    async def _dispatch_claude_sdk(self, meta_prompt: str) -> OOBResult:
        t0 = time.monotonic()
        try:
            from claude_code_sdk import query, ClaudeCodeOptions

            opts = ClaudeCodeOptions(max_turns=8, cwd="/tmp", permission_mode="acceptEdits")
            opts.allowed_tools = [
                "mcp__agent__agent_create_task",
                "mcp__agent__agent_submit_task",
                "mcp__agent__agent_get_task",
                "mcp__agent__agent_get_outputs",
                "mcp__agent__agent_download_file",
            ]

            messages: list[Any] = []
            async for msg in query(prompt=meta_prompt, options=opts):
                messages.append(msg)

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["claude"] = 0
                return OOBResult(backend="claude", status="success", code=code, duration_s=duration)
            return OOBResult(backend="claude", status="error", error="No code", duration_s=duration)
        except Exception as exc:
            return OOBResult(backend="claude", status="error", error=str(exc), duration_s=time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_backend(session_history: list[dict[str, Any]]) -> str:
        scores: dict[str, float] = {}
        for entry in session_history:
            b = entry.get("backend", "")
            if entry.get("outcome") == "PASS":
                scores[b] = scores.get(b, 0) + 10
            elif entry.get("outcome") in ("COMPILE_FAIL", "CORRECTNESS_FAIL"):
                scores[b] = scores.get(b, 0) - 1
            elif entry.get("outcome") == "REGRESSION":
                scores[b] = scores.get(b, 0) + 2
        if not scores:
            return "claude"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    @staticmethod
    def _extract_code_from_messages(messages: list[Any]) -> str | None:
        for m in reversed(messages):
            text = getattr(m, "content", "") if hasattr(m, "content") else ""
            if isinstance(text, str):
                code = OOBBackends._extract_code_block(text)
                if code:
                    return code
        return None

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        if "```" in text:
            parts = text.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith("python"):
                    block = block[6:]
                elif block.startswith("triton"):
                    block = block[6:]
                block = block.strip()
                if len(block) > 50:
                    return block
        return None
