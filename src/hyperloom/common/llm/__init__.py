# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.common.llm`` — shared LLM HTTP protocol adapters.

Protocol layer only: minimal, credential-agnostic POST-and-parse helpers for
the OpenAI-compatible chat-completions API and the Anthropic Messages API, plus
a common error type. It does not decide which env var supplies an API key or
base URL; callers resolve their own credentials and pass ``base_url``/``api_key``
in explicitly. Stdlib + lazy ``httpx`` only, so any package may depend on this
without creating an import cycle.
"""

from __future__ import annotations
