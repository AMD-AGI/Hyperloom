"""Stdio MCP server for upstream PR retrieval and local inclusion checks.

Prefixed tool names avoid collisions, and the REST client enforces request
budgets. ``pr_check_inclusion`` reads local git only and needs no service.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from kernelforge.knowledge.pr_knowledge.agent_tools import TOOL_NAMES
from kernelforge.knowledge.pr_knowledge.client import PRMonitorClient
from kernelforge.knowledge.pr_knowledge.inclusion import probe_inclusion
from kernelforge.knowledge.pr_knowledge.search import PRLifecycle, discover
from kernelforge.knowledge.pr_knowledge.context import PRQueryContext
from kernelforge.tools import mcp_stdio

#: Re-exported so the two servers stay one import away from the shared wire
#: format; both spellings name the same objects.
InvalidParamsError = mcp_stdio.InvalidParamsError

SERVER_NAME = "kernelforge-pr-monitor"

__all__ = ["SERVER_NAME", "TOOL_DEFINITIONS", "TOOL_NAMES", "InvalidParamsError", "handle_tool_call", "main"]

# One file's diff can be enormous; cap what enters the agent's context.
MAX_PATCH_BYTES = 20_000
MAX_FILES_LISTED = 40

# Every tool about one pull request takes its number under the same argument.
_PR_NUMBER = "number"
_PR_NUMBER_SCHEMA = {"type": "integer"}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "pr_find_references",
        "description": (
            "Find upstream pull requests related to a file path or to short "
            "keyword phrases. Returns ranked references with their state "
            "(open, closed without merging, merged, or null when unreported), "
            "a worth_trying score and the distilled optimization summary. Pass "
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
            "Fetch one pull request: title, state, draft flag, merge time, base "
            "branch, changed-file list, commit count and its distilled "
            "summary; a field the service does not report is null. "
            "Use after pr_find_references to see which files a PR touched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                _PR_NUMBER: _PR_NUMBER_SCHEMA,
                "repo": {"type": "string"},
            },
            "required": [_PR_NUMBER],
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
                _PR_NUMBER: _PR_NUMBER_SCHEMA,
                "file_path": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": [_PR_NUMBER, "file_path"],
        },
    },
    {
        "name": "pr_check_inclusion",
        "description": (
            "Check from local git history, with no network, whether an upstream "
            "pull request is already in a source tree. Returns inclusion, the "
            "method that decided it and the evidence. included means a commit "
            "reachable from the checked commit has a subject that names the PR: "
            "a trailing (#N), as a squash merge or a cherry-pick of one writes it, "
            "or 'Merge pull request #N from ...'; or, when you pass "
            "merge_commit_sha (the PR's merge or squash commit upstream), that "
            "every commit it brought in is there by sha or patch-id. Anything "
            "else is unknown, never absent: a backport that was edited, renamed "
            "or folded into another commit looks like no backport at all, so look "
            "for the change itself before treating the PR as missing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                _PR_NUMBER: _PR_NUMBER_SCHEMA,
                "tree": {"type": "string", "description": "A directory inside the git work tree to check."},
                "commit": {"type": "string", "description": "Revision to check; defaults to the tree's HEAD."},
                "merge_commit_sha": {"type": "string", "description": "The PR's merge or squash commit upstream."},
            },
            "required": [_PR_NUMBER, "tree"],
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


def _pr_number(arguments: dict[str, Any]) -> int:
    """Read the required PR number of one call."""
    try:
        return int(arguments[_PR_NUMBER])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidParamsError("number must be an integer") from error


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
                **ref.lifecycle.to_record(),
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
    number = _pr_number(arguments)
    detail = _client().get_pr_detail(repo, number)
    if detail is None:
        return {"repo": repo, "number": number, "reason": "not_found"}
    summary, distill, files, commits = detail.summary, detail.distill, detail.files, detail.commits
    return {
        "repo": repo,
        "number": number,
        "title": summary.get("title", ""),
        **PRLifecycle.from_summary(summary).to_record(),
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
    number = _pr_number(arguments)
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


def _check_inclusion(arguments: dict[str, Any]) -> dict[str, Any]:
    """Probe one tree's local git history for one PR; the commit defaults to the tree's HEAD."""
    number = _pr_number(arguments)
    tree = str(arguments.get("tree") or "").strip()
    if not tree or not Path(tree).is_dir():
        raise InvalidParamsError("tree must name an existing directory")
    commit = str(arguments.get("commit") or "").strip() or "HEAD"
    verdict = probe_inclusion(
        tree=Path(tree),
        commit=commit,
        number=number,
        merge_commit_sha=str(arguments.get("merge_commit_sha") or "").strip() or None,
    )
    return {"number": number, "tree": tree, "commit": commit, **verdict.to_record()}


_HANDLERS = {
    "pr_find_references": _find_references,
    "pr_get_reference": _get_reference,
    "pr_get_file_patch": _get_file_patch,
    "pr_check_inclusion": _check_inclusion,
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
    return await mcp_stdio.dispatch_envelope(
        method,
        params,
        server_name=SERVER_NAME,
        tool_definitions=TOOL_DEFINITIONS,
        handle_tool_call=handle_tool_call,
    )


async def _serve() -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    await mcp_stdio.serve(_dispatch)


def main() -> None:
    """Run the PR Monitor MCP server over standard input and output."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
