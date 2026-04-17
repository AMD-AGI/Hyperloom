"""One minimal Claude Code SDK call to verify ANTHROPIC_API_KEY from TBO .env.

Run from inference_optimization:
  python -m marathon_harness.smoke_test_llm

Does not print secrets.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


async def _run() -> int:
    # Ensure marathon_harness is importable
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from marathon_harness.marathon import _apply_env_to_process, _load_env, _resolve_env_file

    env_path = _resolve_env_file("")
    env = _load_env(env_path)
    _apply_env_to_process(env)

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("FAIL: ANTHROPIC_API_KEY not set after loading .env")
        return 1

    from claude_code_sdk import ClaudeCodeOptions, query

    opts = ClaudeCodeOptions(
        model="claude-sonnet-4-20250514",
        max_turns=3,
        cwd=os.getcwd(),
        system_prompt=None,
        permission_mode="acceptEdits",
    )
    prompt = (
        "Reply with exactly one word: OK. No tools, no files, no explanation."
    )
    messages = []
    async for msg in query(prompt=prompt, options=opts):
        if msg is not None:
            messages.append(msg)

    if not messages:
        print("FAIL: no messages from SDK")
        return 1

    text_parts = []
    for m in messages:
        if hasattr(m, "content") and isinstance(m.content, str):
            text_parts.append(m.content)
    text = "\n".join(text_parts).strip()
    print(f"OK: SDK returned ({len(messages)} msgs, {len(text)} chars)")
    print(f"Preview: {text[:200]!r}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
