# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI Chat Completions tool schemas for ``web_search`` and ``web_fetch``.

The shapes mirror Primus-Claw's ``getToolSchemas`` so the LLM sees the
same surface regardless of whether it talks to Brain or directly to
critic-agent. Descriptions are kept terse — the long-form usage guidance
lives in the system prompt (see ``inference_optimizer/orchestrator/
system_prompts/critic.md``).
"""

from __future__ import annotations

from typing import Any

from .config import IMPLEMENTED_SEARCH_PROVIDERS, WebToolsConfig


def build_tool_schemas(config: WebToolsConfig) -> list[dict[str, Any]]:
    """Build the tool list to pass as ``tools=...`` to OpenAI.

    Tools are gated by config **and** by whether a corresponding client
    would actually be constructible (e.g. search requires a provider with
    an API key). The caller is expected to skip the ``tools=`` argument
    entirely when this list is empty.

    Args:
        config (WebToolsConfig): Resolved web-tools configuration.

    Returns:
        list[dict[str, Any]]: Enabled OpenAI tool schemas (possibly empty).
    """
    if not config.critic_web_tools_enabled:
        return []

    out: list[dict[str, Any]] = []
    if _search_usable(config):
        out.append(_search_schema())
    if config.fetch_enabled:
        out.append(_fetch_schema())
    return out


def _search_usable(config: WebToolsConfig) -> bool:
    """True when at least one implemented search provider has an API key.

    Args:
        config (WebToolsConfig): Resolved web-tools configuration.

    Returns:
        bool: Whether the ``web_search`` tool should be exposed.
    """
    return any(
        name in IMPLEMENTED_SEARCH_PROVIDERS and config.has_search_api_key(name)
        for name in config.search_provider_chain()
    )


def _search_schema() -> dict[str, Any]:
    """Return the OpenAI function schema for the ``web_search`` tool.

    Returns:
        dict[str, Any]: The ``web_search`` tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date information about a fact "
                "you cannot confidently answer from training data. Use to "
                "verify whether a proposal aligns with the current "
                "sglang / vLLM / framework API or is a known regression. "
                "Returns a list of links; you MUST cite every source you "
                "rely on as a markdown hyperlink in notes / advice."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (minimum 2 characters).",
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Restrict results to these domains "
                            "(e.g. ['github.com', 'docs.sglang.ai']). "
                            "Mutually exclusive with blocked_domains."
                        ),
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Exclude results from these domains. Mutually "
                            "exclusive with allowed_domains."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "1-10 (default 5).",
                    },
                    "site": {
                        "type": "string",
                        "description": (
                            "Shorthand for allowed_domains=[site]. Ignored "
                            "if allowed_domains is non-empty."
                        ),
                    },
                    "freshness": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year", "any"],
                        "description": (
                            "Bias toward recency. Honored by 3rd-party "
                            "providers; may be ignored by others."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }


def _fetch_schema() -> dict[str, Any]:
    """Return the OpenAI function schema for the ``web_fetch`` tool.

    Returns:
        dict[str, Any]: The ``web_fetch`` tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a single URL and return its content as Markdown. "
                "Use AFTER web_search picks a promising hit when the "
                "snippet is insufficient. Does NOT execute JavaScript; "
                "for JS-rendered pages a browser MCP tool is required. "
                "Only same-host redirects are followed."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1024,
                        "maximum": 10485760,
                        "description": "HTTP body cutoff (default 10 MiB).",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": (
                            "true -> bypass HTML->Markdown conversion "
                            "(default false)."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    }


__all__ = ["build_tool_schemas"]
