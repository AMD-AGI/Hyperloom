"""GitHub PR search and code search for Hyperloom agents.

Enables automated discovery of optimization techniques from:
  - ROCm/aiter, ROCm/vllm, sgl-project/sglang
  - vllm-project/vllm, NVIDIA/TensorRT-LLM, triton-lang/triton
  - ROCm/composable_kernel, ROCm/rccl

Uses GitHub REST API v3. Requires GITHUB_TOKEN for higher rate limits.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


DEFAULT_REPOS = [
    repo.strip() for repo in
    os.environ.get("HYPERLOOM_SEARCH_REPOS", "").split(",")
    if repo.strip()
] or [
    "ROCm/aiter",
    "ROCm/vllm",
    "sgl-project/sglang",
    "vllm-project/vllm",
    "NVIDIA/TensorRT-LLM",
    "triton-lang/triton",
    "ROCm/hip",
    "ROCm/rccl",
    "ROCm/composable_kernel",
]

GITHUB_API = "https://api.github.com"

_TECHNICAL_TERMS = frozenset({
    "gemm", "moe", "attention", "allreduce", "fp8", "fp16", "bf16", "int8", "int4",
    "quantization", "triton", "ck", "composable_kernel", "aiter",
    "cudagraph", "cuda_graph", "flashattention", "flash_attn",
    "paged_attention", "rope", "rotary", "kv_cache",
    "speculative", "spec_decode", "tensor_parallel", "tp",
    "fused", "fusion", "kernel", "hipify", "rocm", "hip", "nccl", "rccl",
    "allgather", "reducescatter", "reduce_scatter", "all_reduce",
    "prefill", "decode", "batching", "continuous_batching", "chunked_prefill",
    "vllm", "sglang", "trtllm", "tensorrt", "lora", "awq", "gptq",
    "marlin", "w4a16", "w8a8", "smoothquant",
    "custom_all_reduce", "scheduler", "mla", "nsa", "mtp",
})


@dataclass
class PR:
    repo: str
    number: int
    title: str
    author: str
    state: str
    merged_at: str | None
    labels: list[str]
    url: str
    body_snippet: str = ""


@dataclass
class PRDiff:
    repo: str
    number: int
    files_changed: int
    additions: int
    deletions: int
    diff_text: str = ""


@dataclass
class CodeSearchResult:
    repo: str
    path: str
    url: str
    snippet: str = ""


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _api_get(url: str, params: dict | None = None) -> Any:
    """Make a GET request to GitHub API with rate limit handling."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers=_headers())

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 403 and "rate limit" in str(e.read()).lower():
                wait = min(60 * (2 ** attempt), 300)
                time.sleep(wait)
                continue
            if e.code == 422:
                return None
            raise
        except urllib.error.URLError:
            if attempt < 2:
                time.sleep(2)
                continue
            raise
    return None


def search_prs(
    query: str,
    repos: list[str] | None = None,
    state: str = "closed",
    max_results: int = 20,
) -> list[PR]:
    """Search for PRs matching a query across repos."""
    target_repos = repos or DEFAULT_REPOS
    all_prs: list[PR] = []

    for repo in target_repos:
        q = f"{query} repo:{repo} is:pr"
        if state:
            q += f" is:{state}"

        params = {"q": q, "sort": "updated", "order": "desc", "per_page": min(max_results, 30)}
        data = _api_get(f"{GITHUB_API}/search/issues", params)
        if not data or "items" not in data:
            continue

        for item in data["items"][:max_results]:
            body = item.get("body", "") or ""
            all_prs.append(PR(
                repo=repo,
                number=item["number"],
                title=item["title"],
                author=item.get("user", {}).get("login", ""),
                state=item.get("state", ""),
                merged_at=item.get("pull_request", {}).get("merged_at"),
                labels=[l["name"] for l in item.get("labels", [])],
                url=item["html_url"],
                body_snippet=body[:500],
            ))

        if len(all_prs) >= max_results:
            break

    return all_prs[:max_results]


def get_pr_diff(repo: str, pr_number: int, max_size: int = 50000) -> PRDiff | None:
    """Fetch the diff for a specific PR."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    pr_data = _api_get(url)
    if not pr_data:
        return None

    headers = _headers()
    headers["Accept"] = "application/vnd.github.v3.diff"
    req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        diff_text = resp.read().decode("utf-8", errors="replace")[:max_size]
    except Exception:
        diff_text = ""

    return PRDiff(
        repo=repo,
        number=pr_number,
        files_changed=pr_data.get("changed_files", 0),
        additions=pr_data.get("additions", 0),
        deletions=pr_data.get("deletions", 0),
        diff_text=diff_text,
    )


def search_code(
    query: str,
    repos: list[str] | None = None,
    language: str | None = None,
    max_results: int = 10,
) -> list[CodeSearchResult]:
    """Search code across repos."""
    target_repos = repos or DEFAULT_REPOS
    results: list[CodeSearchResult] = []

    repo_qualifier = " ".join(f"repo:{r}" for r in target_repos[:5])
    q = f"{query} {repo_qualifier}"
    if language:
        q += f" language:{language}"

    params = {"q": q, "per_page": min(max_results, 30)}
    data = _api_get(f"{GITHUB_API}/search/code", params)
    if not data or "items" not in data:
        return []

    for item in data["items"][:max_results]:
        results.append(CodeSearchResult(
            repo=item.get("repository", {}).get("full_name", ""),
            path=item.get("path", ""),
            url=item.get("html_url", ""),
        ))

    return results


def extract_keywords(text: str) -> list[str]:
    """Extract technical keywords from text for search queries."""
    words = set(re.findall(r"[a-z][a-z0-9_]+", text.lower()))
    return sorted(words & _TECHNICAL_TERMS)


def build_search_queries(
    model_name: str,
    architecture_keywords: list[str] | None = None,
    bottleneck_keywords: list[str] | None = None,
) -> list[str]:
    """Build a set of search queries for finding optimization PRs."""
    queries = []

    if architecture_keywords:
        for kw in architecture_keywords[:5]:
            queries.append(f"{kw} optimization performance")

    if bottleneck_keywords:
        for kw in bottleneck_keywords[:5]:
            queries.append(f"{kw} improve faster")

    model_short = model_name.split("/")[-1].lower()
    if model_short:
        queries.append(f"{model_short} inference optimization")

    queries.append("performance throughput latency optimization")

    return queries
