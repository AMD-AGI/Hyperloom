"""Framework Agent sibling-skill subpackage.

Driven by ``fa agent`` subcommand (registered in
:mod:`framework_agent.runtime.cli`). Two-stage subprocess protocol used
by ``inference_optimizer``'s ``FrameworkAgentBackend``:

* ``fa agent prepare-task --task task.json --output-bundle bundle.json``
  -- Stage A. Reads the task descriptor from the coordinator, runs AST
  scan when enabled, packages the LLM bundle (prompt + ast_findings +
  KB priors) to ``output-bundle``.
* ``fa agent commit-result --envelope envelope.json --task-id <id>``
  -- Stage B. Validates the LLM-produced envelope against the §4.6
  jsonschema, persists it under ``runs/framework/<task_id>/``, and
  echoes it to stdout for the coordinator to consume.

Submodules:

* :mod:`envelope` -- RESPONSE schemas (4 variants) + jsonschema validator
* :mod:`cli`      -- subprocess entry point (prepare-task / commit-result)
* :mod:`source_resolver` (PR-E) -- vllm/sglang source-root resolution
* :mod:`ast_scanner` (PR-E)     -- libcst-based flag discovery
* :mod:`grep_scanner` (PR-E)    -- per-file grep fallback
* :mod:`flag_discovery` (PR-E)  -- DiscoveredFlag normalisation
* :mod:`patch_proposer` (PR-G)  -- LLM-loop diff generation
* :mod:`kb_priors` (PR-G)       -- framework_optimization partition read
* :mod:`kb_write` (PR-I)        -- KEEP-only lesson append
"""

from __future__ import annotations

__all__: list[str] = []
