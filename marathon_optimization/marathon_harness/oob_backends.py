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
import re
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

# Hard wall-clock for a single _dispatch_*_sdk() call.  The inner Claude
# driving the MCP polling has no notion of wall-clock beyond what the
# meta_prompt tells it, so we wrap the whole query() in asyncio.wait_for.
# Give it the poll-loop timeout plus 120s of slack for create/submit/
# get_outputs/download overhead.
SDK_WRAP_SLACK_S = 120
GEAK_SDK_WRAP_TIMEOUT_S   = GEAK_POLL_TIMEOUT_MIN * 60 + SDK_WRAP_SLACK_S
CODEX_SDK_WRAP_TIMEOUT_S  = CODEX_TIMEOUT_MIN     * 60 + SDK_WRAP_SLACK_S
CLAUDE_SDK_WRAP_TIMEOUT_S = CLAUDE_TIMEOUT_MIN    * 60 + SDK_WRAP_SLACK_S

# Max bytes for a prompt sent to OOB.  Larger prompts are almost always a
# sign that the caller tried to cram multi-file / full-model context into
# a single-kernel optimiser.
OOB_PROMPT_MAX_BYTES = 8 * 1024

# Terminal status strings the inner Claude is told to watch for.  When it
# sees ANY of these from agent_get_task / geak_get_task it MUST return
# immediately instead of continuing to poll.  This is the whole-list
# string — keep it in sync with kernel-manager/SKILL.md IR-12.
OOB_TERMINAL_FAIL_STATES = (
    "failed | cancelled | canceled | error | errored | terminated | "
    "crashed | timeout | timed_out | exhausted | aborted | hw_error | "
    "oom | killed"
)

# Prompt hygiene — patterns that indicate a pollution target (asking
# OOB to do things it physically cannot: multi-GPU, full-model inference,
# end-to-end serving benchmarks, server launch).  Each entry is a plain
# regex, ORed together.  Match = abort dispatch with prompt-pollution.
# See also kernel-manager/actions/dispatch.md §"Prompt Hygiene Guard".
_POLLUTION_PATTERNS = [
    # Inference-server launch
    r"\bvllm\s+serve\b",
    r"sglang[\w\.]*\.?launch_server",
    r"python\s+-m\s+vllm\.entrypoints",
    r"\buvicorn\b",
    r"\btrtllm-serve\b",
    # Multi-GPU / distributed
    r"tensor[-_ ]parallel[-_ ]size\s*[=:]?\s*[2-9]",
    r"--tp[ =][2-9]",
    r"torchrun\s+[^\n]*--nproc[-_]per[-_]node\s*[=]?\s*[2-9]",
    r"\bNCCL_[A-Z_]+",
    r"\bRCCL_[A-Z_]+",
    # End-to-end serving benchmark
    r"\bbenchmark_serving\b",
    r"\bsharegpt\b",
    r"--num-prompts\s+\d{3,}",
    r"\bTTFT\b",
    r"\bTPOT\b",
    r"end[- ]to[- ]end\s+(benchmark|throughput|latency)",
    # Full-model weight loading
    r"--model\s+[^ \n]+\.safetensors",
    r"MODEL_PATH\s*=\s*[^\s]+/[A-Za-z0-9_.-]+-\d+B",
]
_POLLUTION_RE = re.compile("|".join(_POLLUTION_PATTERNS), re.IGNORECASE)


def _scan_prompt_for_pollution(prompt: str) -> str | None:
    """Return None if the prompt is safe for OOB; else a short reason string."""
    if len(prompt.encode("utf-8", errors="ignore")) > OOB_PROMPT_MAX_BYTES:
        return f"prompt-too-large ({len(prompt)} bytes > {OOB_PROMPT_MAX_BYTES})"
    m = _POLLUTION_RE.search(prompt)
    if m is not None:
        return f"banned-pattern: {m.group(0)!r}"
    return None


# Shared polling instructions injected into every meta_prompt.  The inner
# Claude used to only treat "completed" as terminal, which meant a task
# returning "error"/"timeout" would keep it polling until the round-level
# asyncio timeout (up to 60 min) fired for nothing.  These instructions
# force an immediate fail-fast exit on every terminal status.
_OOB_POLL_FAIL_FAST_BLOCK = (
    "FAIL-FAST POLLING RULES (mandatory — violation = wasted marathon time):\n"
    f"  * If agent_get_task / geak_get_task returns status ∈ {{ {OOB_TERMINAL_FAIL_STATES} }},\n"
    "    STOP polling immediately and return the string \"error: <status>: <any message>\".\n"
    "    Do NOT sleep, do NOT retry, do NOT resubmit.\n"
    "  * If the same status is returned 5 times in a row AND it is not in the\n"
    "    active set {running, pending, queued, in_progress, starting, scheduling,\n"
    "    initializing}, call the cancel tool and return \"error: unknown-status-stuck: <status>\".\n"
    "  * If status is \"completed\" / \"succeeded\" / \"done\" / \"finished\", fetch outputs ONCE.\n"
    "    If outputs are empty or download fails, return \"error: empty-output\". Do NOT retry.\n"
    "  * Never write or suggest shell commands that launch an inference server,\n"
    "    run an end-to-end benchmark, or require multi-GPU / full-model weights.\n"
)

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
        # Prompt hygiene pre-check.  OOB backends run a single-kernel
        # optimiser on 1 GPU with no model weights — asking them to
        # launch vllm / run ShareGPT bench / do multi-GPU work will
        # stall the pod for 30 min and return nothing.  Reject such
        # prompts locally so the round-level timeout never fires.
        #
        # Do NOT bump failure_counts here: pollution is an upstream
        # prompt-construction bug (orchestrator gave an over-wide target,
        # or session_history leaked banned words), not a backend fault.
        # Counting it against the backend would trip the
        # BACKEND_FAILURE_THRESHOLD=5 and evict the backend from use on
        # unrelated follow-up targets.
        pollution = _scan_prompt_for_pollution(prompt)
        if pollution is not None:
            log.warning(
                "OOB dispatch aborted (backend=%s, kernel=%s) — prompt-pollution: %s",
                backend, target.get("kernel_name", "?"), pollution,
            )
            return OOBResult(
                backend=backend, status="error",
                error=f"prompt-pollution: {pollution}",
            )

        try:
            if backend == "geak":
                return await self._dispatch_geak(prompt, target, files)
            elif backend == "codex":
                return await self._dispatch_codex(prompt, target)
            elif backend == "claude":
                result = await self._dispatch_claude(prompt, target)
                # Only fall back to Codex on genuine backend errors.
                # prompt-pollution is upstream — Codex will hit the same
                # regex and fail after another round-trip; short-circuit.
                # Likewise on hard SDK timeout: if Claude's 15-min wrap
                # fired, the underlying MCP is misbehaving and Codex
                # won't do better within the caller's round budget.
                if (
                    result.status == "error"
                    and not (result.error or "").startswith("prompt-pollution:")
                ):
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
    # SDK query helper with a hard wall-clock timeout.
    #
    # The inner claude_code_sdk.query() is an async generator driving a
    # sub-Claude that polls an MCP.  If the sub-Claude stalls (e.g. the
    # MCP is hanging on a task that will never complete), the generator
    # would iterate forever; prior to this helper the only safety net was
    # the round-level 3600s timeout in kernel_manager.py, which happily
    # burned 60 min per dead call.  Wrap it in asyncio.wait_for so a
    # single dispatch cannot exceed the poll-loop timeout plus a small
    # slack for create/submit/get_outputs/download overhead.
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_sdk_query(
        meta_prompt: str, opts: Any, timeout_s: float, backend_label: str,
    ) -> list[Any]:
        from claude_code_sdk import query

        async def _drain() -> list[Any]:
            collected: list[Any] = []
            async for msg in query(prompt=meta_prompt, options=opts):
                collected.append(msg)
            return collected

        try:
            return await asyncio.wait_for(_drain(), timeout=timeout_s)
        except asyncio.TimeoutError:
            log.warning(
                "SDK query for %s hit hard timeout %.0fs — abandoning",
                backend_label, timeout_s,
            )
            raise

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
            from claude_code_sdk import ClaudeCodeOptions

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

            messages = await self._run_sdk_query(
                geak_prompt, opts, GEAK_SDK_WRAP_TIMEOUT_S, "geak",
            )

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["geak"] = 0
                return OOBResult(backend="geak", status="success", code=code, duration_s=duration)
            return OOBResult(backend="geak", status="error", error="No code in GEAK output", duration_s=duration)
        except asyncio.TimeoutError:
            return OOBResult(
                backend="geak", status="timeout",
                error=f"SDK hard timeout after {GEAK_SDK_WRAP_TIMEOUT_S}s",
                duration_s=time.monotonic() - t0,
            )
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
            f"{_OOB_POLL_FAIL_FAST_BLOCK}\n"
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
            f"{_OOB_POLL_FAIL_FAST_BLOCK}\n"
            f"--- PROMPT ---\n{prompt}"
        )

        if self.claw_url:
            return await self._dispatch_via_claw(meta_prompt, "codex")

        return await self._dispatch_codex_sdk(meta_prompt)

    async def _dispatch_codex_sdk(self, meta_prompt: str) -> OOBResult:
        t0 = time.monotonic()
        try:
            from claude_code_sdk import ClaudeCodeOptions

            opts = ClaudeCodeOptions(max_turns=8, cwd="/tmp", permission_mode="acceptEdits")
            opts.allowed_tools = [
                "mcp__agent__agent_create_task",
                "mcp__agent__agent_submit_task",
                "mcp__agent__agent_get_task",
                "mcp__agent__agent_get_outputs",
                "mcp__agent__agent_download_file",
                "mcp__agent__agent_cancel_task",
            ]

            messages = await self._run_sdk_query(
                meta_prompt, opts, CODEX_SDK_WRAP_TIMEOUT_S, "codex",
            )

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["codex"] = 0
                return OOBResult(backend="codex", status="success", code=code, duration_s=duration)
            return OOBResult(backend="codex", status="error", error="No code", duration_s=duration)
        except asyncio.TimeoutError:
            return OOBResult(
                backend="codex", status="timeout",
                error=f"SDK hard timeout after {CODEX_SDK_WRAP_TIMEOUT_S}s",
                duration_s=time.monotonic() - t0,
            )
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
            f"{_OOB_POLL_FAIL_FAST_BLOCK}\n"
            f"--- PROMPT ---\n{prompt}"
        )

        if self.claw_url:
            return await self._dispatch_via_claw(meta_prompt, "claude")

        return await self._dispatch_claude_sdk(meta_prompt)

    async def _dispatch_claude_sdk(self, meta_prompt: str) -> OOBResult:
        t0 = time.monotonic()
        try:
            from claude_code_sdk import ClaudeCodeOptions

            opts = ClaudeCodeOptions(max_turns=8, cwd="/tmp", permission_mode="acceptEdits")
            opts.allowed_tools = [
                "mcp__agent__agent_create_task",
                "mcp__agent__agent_submit_task",
                "mcp__agent__agent_get_task",
                "mcp__agent__agent_get_outputs",
                "mcp__agent__agent_download_file",
                "mcp__agent__agent_cancel_task",
            ]

            messages = await self._run_sdk_query(
                meta_prompt, opts, CLAUDE_SDK_WRAP_TIMEOUT_S, "claude",
            )

            code = self._extract_code_from_messages(messages)
            duration = time.monotonic() - t0
            if code:
                self.failure_counts["claude"] = 0
                return OOBResult(backend="claude", status="success", code=code, duration_s=duration)
            return OOBResult(backend="claude", status="error", error="No code", duration_s=duration)
        except asyncio.TimeoutError:
            return OOBResult(
                backend="claude", status="timeout",
                error=f"SDK hard timeout after {CLAUDE_SDK_WRAP_TIMEOUT_S}s",
                duration_s=time.monotonic() - t0,
            )
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
