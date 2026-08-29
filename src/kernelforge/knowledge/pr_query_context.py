"""Build repository, path, and keyword inputs for PR Monitor queries.

Repository resolution must succeed before discovery can run.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

# Query outcomes.
REASON_REPO_UNRESOLVED = "repo_unresolved"
REASON_REPO_UNTRACKED = "repo_untracked"
REASON_NO_CANDIDATE = "no_candidate"
REASON_SKIPPED_DEADLINE = "skipped_deadline"
REASON_SERVICE_UNREACHABLE = "service_unreachable"
REASON_CONTRACT_ERROR = "contract_error"
# The subsystem itself failed locally (filesystem, parsing) rather than the
# service being unavailable.
REASON_LOCAL_FAILURE = "local_failure"

# Repositories the service is expected to track. Drift against /repos is worth
# a warning because it means the server-side ConfigMap moved.
PR_REPOS_EXPECTED: tuple[str, ...] = (
    "ROCm/aiter",
    "ROCm/ATOM",
    "ROCm/FlyDSL",
    "ROCm/hip",
    "ROCm/vllm",
    "sgl-project/sglang",
    "triton-lang/triton",
    "vllm-project/vllm",
)

# Known repositories that are not expected to be indexed yet.
PR_REPOS_WISHLIST: tuple[str, ...] = (
    "NVIDIA/nccl",
    "NVIDIA/TensorRT-LLM",
    "pytorch/pytorch",
    "ROCm/rccl",
    "ROCm/ROCm",
)

# Kernel backends with an exact repository mapping.
KERNEL_BACKEND_REPO_MAP: dict[str, str] = {
    "aiter": "ROCm/aiter",
    "flydsl": "ROCm/FlyDSL",
    "triton": "triton-lang/triton",
    # Gluon ships inside Triton (``triton.experimental.gluon``), so its PRs,
    # its breakage and its API churn all live in the same repository.
    "gluon": "triton-lang/triton",
    "hip": "ROCm/hip",
}

# Forks whose upstream carries the interesting history.
FORK_UPSTREAM_MAP: dict[str, str] = {
    "ROCm/vllm": "vllm-project/vllm",
}

_MIN_TOKEN_LEN = 3
_MAX_KEYWORDS = 4
# Terms too broad to narrow a kernel PR search.
_STOPWORDS = frozenset(
    {
        "and",
        "block",
        "code",
        "cpp",
        "cuda",
        "fix",
        "for",
        "function",
        "gpu",
        "hip",
        "impl",
        "kernel",
        "kernels",
        "not",
        "src",
        "support",
        "test",
        "the",
        "use",
        "util",
        "utils",
        "with",
    }
)


@dataclass(frozen=True)
class PRQueryContext:
    """Everything the discovery pipeline needs, or the reason it cannot run."""

    repo: str = ""
    file_paths: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    reason: str = ""

    @property
    def usable(self) -> bool:
        """True when the pipeline has a repo and at least one query source."""
        return bool(self.repo) and not self.reason and bool(self.file_paths or self.keywords)


@dataclass
class WhitelistDrift:
    """Difference between the expected repository set and what /repos reports."""

    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    inactive: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """True when nothing drifted and every expected repo is active."""
        return not (self.missing or self.unexpected or self.inactive)


def normalize_kernel_backend(kernel_backend: str) -> str:
    """Reduce a backend label to its canonical key."""
    return (kernel_backend or "").strip().lower()


def parse_git_remote(url: str) -> str:
    """Extract ``owner/repo`` from an SSH or HTTPS remote URL."""
    raw = (url or "").strip().removesuffix(".git")
    if not raw:
        return ""
    if "://" not in raw and ":" in raw:
        path = raw.rsplit(":", 1)[-1]
    else:
        segments = raw.split("://", 1)[-1].split("/")
        path = "/".join(segments[1:]) if len(segments) > 1 else ""
    parts = [segment for segment in path.split("/") if segment]
    if len(parts) != 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def resolve_repo(
    *,
    kernel_backend: str = "",
    git_remote: str = "",
    tracked: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Resolve ``owner/repo`` from the kernel backend, remote, and fork map."""
    candidates: list[str] = []
    mapped = KERNEL_BACKEND_REPO_MAP.get(normalize_kernel_backend(kernel_backend))
    if mapped:
        candidates.append(mapped)
    from_remote = parse_git_remote(git_remote)
    if from_remote:
        candidates.append(from_remote)
        upstream = FORK_UPSTREAM_MAP.get(from_remote)
        if upstream:
            candidates.append(upstream)
    if not candidates:
        return "", REASON_REPO_UNRESOLVED
    if tracked is None:
        return candidates[0], ""
    for candidate in candidates:
        if candidate in tracked:
            return candidate, ""
    return candidates[0], REASON_REPO_UNTRACKED


def normalize_file_path(
    raw: str,
    *,
    workspace: str = "",
    exists: Callable[[str], bool] | None = None,
) -> str:
    """Return a verified repo-relative POSIX path, or ``""``."""
    candidate = (raw or "").strip().replace("\\", "/")
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        if not workspace:
            return ""
        root = os.path.abspath(workspace).replace("\\", "/")
        absolute = os.path.abspath(candidate).replace("\\", "/")
        if absolute != root and not absolute.startswith(root.rstrip("/") + "/"):
            return ""
        candidate = os.path.relpath(absolute, root).replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or candidate.startswith("/") or ".." in candidate.split("/"):
        return ""
    if exists is not None and not exists(candidate):
        return ""
    return candidate


def _tokens(text: str) -> list[str]:
    """Split an identifier into ordered lowercase tokens."""
    tokens: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", text or ""):
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", chunk):
            word = piece.lower()
            if len(word) >= _MIN_TOKEN_LEN and not word.isdigit():
                tokens.append(word)
    return tokens


def extract_keywords(
    *,
    operator_name: str = "",
    target_functions: tuple[str, ...] | list[str] = (),
    bottleneck: str = "",
    limit: int = _MAX_KEYWORDS,
) -> tuple[str, ...]:
    """Build one- and two-word phrases for whole-string ILIKE search."""
    phrases: list[str] = []

    def add(phrase: str) -> None:
        """Append a phrase once, preserving discovery order."""
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    for source in (operator_name, *target_functions, bottleneck):
        tokens = _tokens(source)
        # Bigrams use tokens adjacent in the original text. Filtering stopwords
        # first would splice together words that never co-occur ("fused rms"
        # from "fused_add_rms_norm"), and the server matches verbatim.
        for first, second in zip(tokens, tokens[1:]):
            if first in _STOPWORDS and second in _STOPWORDS:
                continue
            add(f"{first} {second}")
        for token in tokens:
            if token not in _STOPWORDS:
                add(token)
    return tuple(phrases[:limit])


def check_whitelist(repos_payload: list[dict]) -> WhitelistDrift:
    """Compare expected repositories with the service's active set."""
    actual = {str(entry.get("repo_name") or "") for entry in repos_payload if isinstance(entry, dict)}
    actual.discard("")
    inactive = {
        str(entry.get("repo_name") or "")
        for entry in repos_payload
        if isinstance(entry, dict) and not entry.get("is_active", True)
    }
    expected = set(PR_REPOS_EXPECTED)
    return WhitelistDrift(
        missing=tuple(sorted(expected - actual)),
        unexpected=tuple(sorted(actual - expected - set(PR_REPOS_WISHLIST))),
        inactive=tuple(sorted(inactive & expected)),
    )


def build_context(
    *,
    kernel_backend: str = "",
    git_remote: str = "",
    tracked: tuple[str, ...] | None = None,
    source_files: tuple[str, ...] | list[str] = (),
    workspace: str = "",
    exists: Callable[[str], bool] | None = None,
    operator_name: str = "",
    target_functions: tuple[str, ...] | list[str] = (),
    bottleneck: str = "",
) -> PRQueryContext:
    """Build the repository, path, and keyword query context."""
    repo, reason = resolve_repo(kernel_backend=kernel_backend, git_remote=git_remote, tracked=tracked)
    # Preserve paths so an untracked fork can be resolved by source ownership.
    paths = []
    for raw in source_files:
        normalized = normalize_file_path(raw, workspace=workspace, exists=exists)
        if normalized and normalized not in paths:
            paths.append(normalized)
    keywords = extract_keywords(
        operator_name=operator_name,
        target_functions=target_functions,
        bottleneck=bottleneck,
    )
    return PRQueryContext(
        repo=repo,
        file_paths=tuple(paths[:3]),
        keywords=keywords,
        reason=reason,
    )
