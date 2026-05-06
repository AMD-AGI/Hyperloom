#!/usr/bin/env python3
"""Filter stream-json claude output into human-readable log lines."""
import json, sys, textwrap

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue

    t = obj.get("type")
    msg = obj.get("message", {})

    if t == "system" and obj.get("subtype") == "init":
        print(f"[INIT] model={obj.get('model')} tools={len(obj.get('tools', []))}")
        sys.stdout.flush()

    elif t == "assistant":
        for c in msg.get("content", []):
            if isinstance(c, dict):
                if c.get("type") == "text":
                    text = c["text"].strip()
                    if text:
                        print(f"\n{text}\n")
                        sys.stdout.flush()
                elif c.get("type") == "tool_use":
                    name = c.get("name", "?")
                    inp = c.get("input", {})
                    if name == "Bash":
                        cmd = inp.get("command", "")[:120]
                        print(f"  > {name}: {cmd}")
                    elif name in ("Read", "Write", "Edit"):
                        path = inp.get("file_path", inp.get("path", ""))
                        print(f"  > {name}: {path}")
                    elif name == "Grep":
                        print(f"  > {name}: '{inp.get('pattern', '')}' in {inp.get('path', '.')}")
                    else:
                        print(f"  > {name}: {str(inp)[:100]}")
                    sys.stdout.flush()

    elif t == "user":
        for c in msg.get("content", []):
            if isinstance(c, dict) and c.get("type") == "tool_result":
                content = c.get("content", "")
                if isinstance(content, str):
                    if content.startswith("<persisted-output>"):
                        print(f"  ← (large output, saved to file)")
                    elif "is_error" in str(c) and c.get("is_error"):
                        print(f"  ← ERROR: {content[:200]}")
                    else:
                        snippet = content[:200].replace("\n", " ").replace("\t", " ")
                        print(f"  ← {snippet}")
                    sys.stdout.flush()

    elif t == "result":
        cost = obj.get("cost_usd")
        dur = obj.get("duration_ms")
        if cost or dur:
            print(f"\n[DONE] cost=${cost:.4f} duration={dur/1000:.1f}s" if cost and dur else f"\n[DONE]")
            sys.stdout.flush()
