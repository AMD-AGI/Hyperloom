---
myst:
  html_meta:
    "description": "How to extend KernelForge: add a new fellow agent, a new MCP tool, or new knowledge-base entries."
    "keywords": "KernelForge, extending, add fellow, add MCP tool, knowledge base, fellows/base.py, mcp_server"
---

# Add a fellow, tool, or knowledge

KernelForge is designed to be extended. The three most common extension points
are a new fellow, a new GPU tool, and new knowledge.

## Add a new fellow agent

1. Create `src/kernel_agents/fellows/mybackend/` with:
   - `__init__.py`
   - `prompts.py` (defining `build_system_prompt(gpu_target, knowledge_content)`)
2. Add backend knowledge under `local_knowledge/languages/mybackend/`.
3. Register the backend in
   `src/kernel_agents/fellows/constants.py:FELLOW_PROMPT_MODULES`.

`--fellow mybackend-fellow` then selects it.

## Add a GPU toolchain helper

1. Create `src/kernel_agents/mcp_server/tools/mytool.py`.
2. Call it from the loop stage that needs it — the tools are plain functions,
   invoked directly rather than through a protocol.

## Add knowledge

Drop a `.md` file into `knowledge_base/<backend>/`. It is automatically loaded
and injected into the relevant fellow's prompt. Keep each file under about
2K tokens so prompts stay focused.
