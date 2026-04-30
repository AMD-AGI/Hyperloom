# Reference — Kernel Agent

Long-tail reference docs the kernel agent reads once-per-session and
remembers across turns.

| File | When to read |
|---|---|
| `geak_guide.md` | Before first `run_optimization` — kernel categories, source paths, GEAK MCP tool semantics |
| `oob_guide.md` | Before invoking the OOB (codex/claude/llm) backend — prompt conventions + Ray submit patterns |
| `ir_soft_rules.md` | Before any kernel-opt work — IR-1/2/6/7 are WARN-only (see SKILL.md for IR-3/4/5 BLOCK) |
| `wire_protocol_quickref.md` | When PROTOCOL.md isn't visible via Read; envelope JSON shape + bash recipe |
| `troubleshooting.md` | When something fails (timeout, OOM, patch crash, accuracy revert) — recovery table |
