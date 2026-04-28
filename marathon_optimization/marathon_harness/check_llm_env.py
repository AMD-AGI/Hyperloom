"""Quick check: .env → CLAW_URL / ANTHROPIC_API_KEY + claude_code_sdk / claude CLI.

Run from TBO repo root:
  python -m marathon_harness.check_llm_env

Or with explicit .env:
  python inference_optimization/marathon_harness/check_llm_env.py /path/to/.env
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / ".env"
    print(f"Checking .env: {env_path}")

    env_vars: dict[str, str] = {}
    if not env_path.is_file():
        print("  (file missing — copy .env.template to .env)")
    else:
        for raw in env_path.read_text().splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[7:].strip()
            if "=" not in s:
                continue
            k, _, v = s.partition("=")
            env_vars[k.strip()] = v.strip().strip("'\"")

    claw = env_vars.get("CLAW_URL", "")
    ak = env_vars.get("ANTHROPIC_API_KEY", "")

    if claw:
        print(f"  CLAW_URL: {claw}  (Primus-Claw backend — no Anthropic key needed)")
        token = env_vars.get("CLAW_AUTH_TOKEN", "")
        print(f"  CLAW_AUTH_TOKEN: {'set' if token else 'empty (dev mode, no auth)'}")
    elif ak:
        print(f"  ANTHROPIC_API_KEY: set (prefix={ak[:12]}… len={len(ak)})")
    else:
        print("  No LLM backend configured — set CLAW_URL or ANTHROPIC_API_KEY")

    spec = importlib.util.find_spec("claude_code_sdk")
    print(f"  claude_code_sdk: {'installed' if spec else 'NOT installed (pip install claude-code-sdk)'}")

    claude = shutil.which("claude")
    print(f"  claude CLI: {claude or 'not on PATH'}")

    if not claw and not ak:
        print("\n  Recommendation: set CLAW_URL to a Primus-Claw instance,")
        print("  or set ANTHROPIC_API_KEY for direct claude_code_sdk usage.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
