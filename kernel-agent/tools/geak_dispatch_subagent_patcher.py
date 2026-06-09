#!/usr/bin/env python3
"""Ensure GEAK dispatch loads general-kernel-optimization subagent prompts.

Zero-touch runs observed ``You are a helpful assistant`` in fixed-canonical
sub-agents because ``SubAgentRegistry()`` resolved ``subagents/`` via
``get_repo_root()``, which points at site-packages when GEAK is not run from
an editable checkout. Setting ``GEAK_ROOT`` fixes lookup; this patch makes
dispatch honour ``GEAK_ROOT/subagents`` explicitly as a belt-and-suspenders fix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SENTINEL = "# HYPERLOOM_GEAK_SUBAGENT_ROOT"

_OLD = """            registry = SubAgentRegistry()
            descriptor = registry.get(agent_name)"""

_NEW = """            subagents_dir = None
            geak_root = os.environ.get("GEAK_ROOT", "").strip()
            if geak_root:
                cand = Path(geak_root).expanduser() / "subagents"
                if cand.is_dir():
                    subagents_dir = cand
            registry = SubAgentRegistry(subagents_dir=subagents_dir)
            descriptor = registry.get(agent_name)"""


def _targets() -> list[Path]:
    paths: list[Path] = []
    mirror = (
        Path(os.environ.get("HYPERLOOM_ROOT", ""))
        / "geak"
        / "src"
        / "minisweagent"
        / "run"
        / "dispatch.py"
    )
    if mirror.is_file():
        paths.append(mirror)
    try:
        import minisweagent.run.dispatch as mod

        paths.append(Path(mod.__file__).resolve())
    except Exception:
        pass
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def patch_file(path: Path) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8")
    if _SENTINEL in text:
        return True, "already patched"
    if _OLD not in text:
        return False, f"anchor not found in {path}"
    text = text.replace(_OLD, _NEW + "\n            " + _SENTINEL, 1)
    path.write_text(text, encoding="utf-8")
    return True, "patched"


def main() -> int:
    ok_any = False
    for path in _targets():
        ok, msg = patch_file(path)
        print(f"[geak-dispatch-subagent] {path}: {msg}")
        ok_any = ok_any or (ok and msg == "patched")
    if not _targets():
        print("[geak-dispatch-subagent] WARN: no dispatch.py targets", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
