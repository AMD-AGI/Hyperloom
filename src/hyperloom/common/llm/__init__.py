# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.common.llm`` — shared LLM HTTP protocol adapters (tree-reform.MD §4/§7).

This package holds the **protocol** layer only: minimal, credential-agnostic
POST-and-parse helpers for the OpenAI-compatible chat-completions API and the
Anthropic Messages API, plus a common error type. It intentionally does NOT
decide which environment variable supplies an API key or base URL — that is
a caller concern (see ``tree-reform.MD`` §12.2: "role-specific 行为...仍留
orchestrator/roles/"). Callers resolve their own credentials and pass
``base_url``/``api_key`` in explicitly.

Zero first-party imports (stdlib + lazy ``httpx``) so any package may depend
on this without creating an import cycle.
"""

from __future__ import annotations
