# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral LLM gateway resolution shared across kernelforge."""

from __future__ import annotations

from .gateway import (
    LlmGateway,
    normalize_anthropic_base_url,
    resolve_anthropic_gateway,
    resolve_openai_gateway,
)

__all__ = [
    "LlmGateway",
    "normalize_anthropic_base_url",
    "resolve_anthropic_gateway",
    "resolve_openai_gateway",
]
