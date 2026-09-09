"""Stdio MCP server for upstream PR retrieval.

Prefixed tool names avoid collisions, and the REST client enforces request
budgets.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from kernelforge.knowledge.pr_monitor_client import (
    PRContractError,
    PRMonitorClient,
)
from kernelforge.knowledge.pr_monitor_search import discover
from kernelforge.knowledge.pr_query_context import PRQueryContext
from kernelforge.mcp_server import stdio_transport

#: Re-exported so the two servers stay one import away from the shared wire
#: format; both spellings name the same objects.
InvalidParamsError = stdio_transport.InvalidParamsError
_write_message = stdio_transport.write_message
_write_error = stdio_transport.write_error

SERVER_NAME = "kernelforge-pr-monitor"
# Agents see these as mcp__pr_monitor__<name>.
TOOL_NAMES = ("pr_find_references", "pr_get_reference", "pr_get_file_patch")

# One file's diff can be enormous; cap what enters the agent's context.
MAX_PATCH_BYTES = 20_000
MAX_FILES_LISTED = 40


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "pr_find_references",
        "description": (
            "Find merged/open upstream pull requests related to a file path or "
            "to short keyword phrases. Returns ranked references with a "
            "worth_trying score and the distilled optimization summary. Pass "
            "one- or two-word phrases: the search matches the whole query "
            "string, so a sentence returns nothing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Repository-relative path, matched exactly.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short phrases, each searched separately.",
                },
                "repo": {
                    "type": "string",
                    "description": "owner/repo; defaults to the campaign's repo.",
                },
            },
        },
    },
    {
        "name": "pr_get_reference",
        "description": (
            "Fetch one pull request: title, state, changed-file list, commit "
            "count and its distilled summary. Use after pr_find_references to "
            "see which files a PR touched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "number": {"type": "integer"},
                "repo": {"type": "string"},
            },
            "required": ["number"],
        },
    },
    {
        "name": "pr_get_file_patch",
        "description": (
            "Fetch the diff of ONE file changed by a pull request. Prefer this "
            "over reading every patch. May report absent if the PR was "
            "force-pushed since the path was indexed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "number": {"type": "integer"},
                "file_path": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": ["number", "file_path"],
        },
    },
]


def _resolve_repo(arguments: dict[str, Any]) -> str:
    """Pick and validate the target repository for one call."""
    repo = str(arguments.get("repo") or os.environ.get("PR_KB_REPO", "")).strip()
    if not repo:
        raise InvalidParamsError("no repo configured for this campaign; pass repo=owner/name")
    parts = [segment for segment in repo.split("/") if segment]
    if len(parts) != 2:
        raise InvalidParamsError(f"repo must be owner/name, got {repo!r}")
    return f"{parts[0]}/{parts[1]}"


def _client() -> PRMonitorClient:
    """Build a client whose own timeouts bound every call."""
    return PRMonitorClient()


def _find_references(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the discovery pipeline for an agent-supplied path and/or keywords."""
    repo = _resolve_repo(arguments)
    keywords = arguments.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    file_path = str(arguments.get("file_path") or "").strip()
    context = PRQueryContext(
        repo=repo,
        file_paths=(file_path,) if file_path else (),
        keywords=tuple(str(word).strip() for word in keywords if str(word).strip()),
    )
    if not context.usable:
        return {"repo": repo, "reason": "no file_path or keywords given", "results": []}
    outcome = discover(_client(), context)
    result = {
        "repo": repo,
        "reason": outcome.reason or "ok",
        "results": [
            {
                "number": ref.number,
                "title": ref.title,
                "state": "merged" if ref.is_merged else "open",
                "worth_trying": ref.worth_trying,
                "hit_via": list(ref.hit_via),
                "components": list(ref.components),
                "mechanisms": list(ref.mechanisms),
                "summary": ref.summary,
                "risk_notes": ref.risk_notes,
                "n_files": ref.n_files,
            }
            for ref in outcome.references
        ],
    }
    if outcome.stats.get("degraded_reason"):
        result["degraded_reason"] = outcome.stats["degraded_reason"]
    return result


def _get_reference(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fetch one PR's metadata, file list and distill in a single hop."""
    repo = _resolve_repo(arguments)
    try:
        number = int(arguments["number"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidParamsError("number must be an integer") from error
    detail = _client().get_pr(repo, number)
    if detail is None:
        return {"repo": repo, "number": number, "reason": "not_found"}
    summary = detail.get("summary")
    if summary is None:
        summary = {}
    if not isinstance(summary, dict):
        raise PRContractError("PR response field 'summary' must be an object")
    distill = detail.get("distill")
    if distill is None:
        distill = {}
    if not isinstance(distill, dict):
        raise PRContractError("PR response field 'distill' must be an object")
    files = detail.get("files")
    if files is None:
        files = []
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        raise PRContractError("PR response field 'files' must be an array of objects")
    commits = detail.get("commits")
    if commits is None:
        commits = []
    if not isinstance(commits, list):
        raise PRContractError("PR response field 'commits' must be an array")
    return {
        "repo": repo,
        "number": number,
        "title": summary.get("title", ""),
        "state": "merged" if summary.get("is_merged") else "open",
        "additions": summary.get("additions"),
        "deletions": summary.get("deletions"),
        "n_files": len(files),
        "files_truncated": len(files) > MAX_FILES_LISTED,
        # List rows use ``file_path``; the by-path query uses ``path``.
        "files": [
            {
                "file_path": item.get("file_path"),
                "status": item.get("status"),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "has_patch": item.get("has_patch"),
                "is_binary": item.get("is_binary"),
            }
            for item in files[:MAX_FILES_LISTED]
        ],
        "commits": len(commits),
        "distill": {
            "status": distill.get("status"),
            "worth_trying": distill.get("worth_trying"),
            "summary": distill.get("summary"),
            "components": distill.get("components"),
            "mechanisms": distill.get("mechanisms"),
            "expected_gain": distill.get("expected_gain"),
            "risk_notes": distill.get("risk_notes"),
        }
        if distill
        else None,
    }


def _get_file_patch(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fetch one file's diff, truncated to a context-safe size."""
    repo = _resolve_repo(arguments)
    try:
        number = int(arguments["number"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidParamsError("number must be an integer") from error
    file_path = str(arguments.get("file_path") or "").strip()
    if not file_path:
        raise InvalidParamsError("file_path must be a non-empty string")
    payload = _client().get_file_patch(repo, number, file_path)
    if payload is None:
        return {
            "repo": repo,
            "number": number,
            "file_path": file_path,
            "reason": "absent_at_current_head",
        }
    if "patch" not in payload:
        raise PRContractError("file-patch response must contain 'patch'")
    patch = str(payload["patch"] or "")
    encoded = patch.encode("utf-8")
    truncated = len(encoded) > MAX_PATCH_BYTES
    if truncated:
        patch = encoded[:MAX_PATCH_BYTES].decode("utf-8", errors="ignore")
    return {
        "repo": repo,
        "number": number,
        "file_path": file_path,
        "truncated": truncated,
        "patch": patch,
    }


_HANDLERS = {
    "pr_find_references": _find_references,
    "pr_get_reference": _get_reference,
    "pr_get_file_patch": _get_file_patch,
}


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke one PR tool off the event loop and wrap it as MCP content."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise InvalidParamsError(f"unknown tool: {name}")
    result = await asyncio.to_thread(handler, arguments)
    return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}


async def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one supported MCP request and return its result object."""
    return await stdio_transport.dispatch_envelope(
        method,
        params,
        server_name=SERVER_NAME,
        tool_definitions=TOOL_DEFINITIONS,
        handle_tool_call=handle_tool_call,
    )


async def _serve() -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    await stdio_transport.serve(_dispatch)


def main() -> None:
    """Run the PR Monitor MCP server over standard input and output."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
