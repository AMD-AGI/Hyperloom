# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed data model for targeted (compiled-component) builds.

A :class:`TargetedBuildAction` describes the one missing *compiled* component
the enablement loop must build (an AITER FP4 MoE / MLA / NSA op, sgl-kernel, or
vLLM from source). A :class:`BuildResult` carries the outcome plus the resolved
:class:`FrameworkRuntime` a KEEP promotes. :func:`build_novelty_key` gives the
repeat-vs-novel identity the stall gate keys on.

Pure Python: no network, subprocess, or filesystem access. The compile itself
lives in ``targeted_build.py``. :class:`FrameworkRuntime` is the shared runtime
contract owned by :mod:`stack_actions`; it is re-exported here so build callers
import one type.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .stack_actions import FrameworkRuntime


# Components a targeted build may acquire.
_COMPONENTS: frozenset[str] = frozenset({"aiter", "sgl_kernel", "vllm_source", "framework_ext"})

# The acquisition-attempt outcome (a second axis from FailureSignature.kind):
# what happened to the *build*, distinguishing "ran out of time" from a defect.
FAILURE_CLASSES: tuple[str, ...] = (
    "ok",
    "preflight_disk",
    "preflight_toolchain",
    "preflight_budget",
    "timeout",
    "abi_mismatch",
    "compile_error",
    "symbol_missing",
    "boot_failed",
    "correctness_failed",
)


def normalize_failure_class(value: Any) -> str:
    """Coerce an arbitrary value to a known failure class (``ok`` fallback)."""
    text = str(value or "").strip()
    return text if text in FAILURE_CLASSES else "ok"


@dataclass(frozen=True)
class TargetedBuildAction:
    """One compiled-component build the enablement loop may run.

    Attributes:
        gap_id: Canonical gap id the build addresses.
        framework: Target framework (``vllm`` / ``sglang``).
        component: One of :data:`_COMPONENTS`.
        capability: The missing capability (``fp4_moe`` / ``mla`` / ``nsa`` / ...).
        reason: Human-readable justification / evidence summary.
        repo_url: Origin-allowlisted git URL.
        ref: Pinned ref; empty means tag-descending autoselect.
        autoselect_tag_glob: Tag glob for autoselect when ``ref`` is empty.
        gpu_arch: EXPLICIT ``gfx942`` / ``gfx950`` (never inferred).
        max_jobs: Parallelism cap; 0 means the per-component default.
        expected_symbols: Symbols the artifact verifier must find.
        expected_artifacts: Artifact globs the verifier must find fresh.
        build_command: argv-only build command (no shell strings).
        torch_constraint_mode: How the ROCm torch pin is applied.
        build_budget_sec: Wall-clock hard timeout; 0 means per-component default.
        server_args: Extra server args routed to ``EXTRA_{FW}_ARGS``.
        envs: Extra structured build env.
        attempt_root: ``$SESSION_DIR/enablement/builds/<attempt_id>``.
        source_pr_url: Source PR URL that drove the ref choice (provenance; does
            not affect :func:`build_novelty_key`).
    """

    gap_id: str
    framework: str
    component: str
    capability: str
    reason: str = ""
    repo_url: str = ""
    ref: str = ""
    autoselect_tag_glob: str = "v*"
    gpu_arch: str = ""
    max_jobs: int = 0
    expected_symbols: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    build_command: tuple[str, ...] = ()
    torch_constraint_mode: Literal["constraint_file", "wheel_index_pin"] = "constraint_file"
    build_budget_sec: int = 0
    server_args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)
    attempt_root: str = ""
    source_pr_url: str = ""

    def to_state(self) -> dict[str, Any]:
        """Serialize to a plain dict for task params / shared state."""
        return {
            "gap_id": self.gap_id,
            "framework": self.framework,
            "component": self.component,
            "capability": self.capability,
            "reason": self.reason,
            "repo_url": self.repo_url,
            "ref": self.ref,
            "autoselect_tag_glob": self.autoselect_tag_glob,
            "gpu_arch": self.gpu_arch,
            "max_jobs": self.max_jobs,
            "expected_symbols": list(self.expected_symbols),
            "expected_artifacts": list(self.expected_artifacts),
            "build_command": list(self.build_command),
            "torch_constraint_mode": self.torch_constraint_mode,
            "build_budget_sec": self.build_budget_sec,
            "server_args": self.server_args,
            "envs": dict(self.envs),
            "attempt_root": self.attempt_root,
            "source_pr_url": self.source_pr_url,
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "TargetedBuildAction":
        """Rehydrate from a plain dict; missing keys default sensibly."""
        d = d or {}

        def _tuple(key: str) -> tuple[str, ...]:
            raw = d.get(key)
            return tuple(str(x) for x in raw) if isinstance(raw, (list, tuple)) else ()

        raw_envs = d.get("envs")
        envs = {str(k): str(v) for k, v in raw_envs.items()} if isinstance(raw_envs, dict) else {}

        component = str(d.get("component") or "")
        if component not in _COMPONENTS:
            component = "aiter"
        mode = str(d.get("torch_constraint_mode") or "constraint_file")
        if mode not in ("constraint_file", "wheel_index_pin"):
            mode = "constraint_file"

        def _int(key: str) -> int:
            try:
                return int(d.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            gap_id=str(d.get("gap_id") or ""),
            framework=str(d.get("framework") or "").strip().lower(),
            component=component,
            capability=str(d.get("capability") or ""),
            reason=str(d.get("reason") or ""),
            repo_url=str(d.get("repo_url") or ""),
            ref=str(d.get("ref") or ""),
            autoselect_tag_glob=str(d.get("autoselect_tag_glob") or "v*"),
            gpu_arch=str(d.get("gpu_arch") or ""),
            max_jobs=_int("max_jobs"),
            expected_symbols=_tuple("expected_symbols"),
            expected_artifacts=_tuple("expected_artifacts"),
            build_command=_tuple("build_command"),
            torch_constraint_mode=mode,  # type: ignore[arg-type]
            build_budget_sec=_int("build_budget_sec"),
            server_args=str(d.get("server_args") or ""),
            envs=envs,
            attempt_root=str(d.get("attempt_root") or ""),
            source_pr_url=str(d.get("source_pr_url") or ""),
        )


@dataclass(frozen=True)
class BuildResult:
    """Outcome of one :class:`TargetedBuildAction`.

    Attributes:
        ok: Whether the build + artifact verify succeeded.
        attempt_root: The attempt directory the build ran in.
        runtime: The resolved runtime a KEEP promotes (empty on failure).
        built_artifacts: Verified artifact paths.
        installed_versions: torch(+hip) / ref-tag+sha / arch, for reproducibility.
        build_log_path: Path to the compile log.
        verify_log_path: Path to the verify log.
        error: Failure reason when ``ok`` is False.
        failure_class: One of :data:`FAILURE_CLASSES` (time vs defect axis).
        failure_summary: Human/agent-readable summary fed to the framework channel.
    """

    ok: bool
    attempt_root: str = ""
    runtime: FrameworkRuntime = field(default_factory=FrameworkRuntime)
    built_artifacts: tuple[str, ...] = ()
    installed_versions: Mapping[str, str] = field(default_factory=dict)
    build_log_path: str = ""
    verify_log_path: str = ""
    error: str = ""
    failure_class: str = "ok"
    failure_summary: str = ""

    def to_state(self) -> dict[str, Any]:
        """Serialize to a plain dict for shared state / observability."""
        return {
            "ok": self.ok,
            "attempt_root": self.attempt_root,
            "runtime": self.runtime.to_state(),
            "built_artifacts": list(self.built_artifacts),
            "installed_versions": dict(self.installed_versions),
            "build_log_path": self.build_log_path,
            "verify_log_path": self.verify_log_path,
            "error": self.error,
            "failure_class": self.failure_class,
            "failure_summary": self.failure_summary,
        }

    @classmethod
    def from_state(cls, d: Mapping[str, Any] | None) -> "BuildResult":
        """Rehydrate from a plain dict."""
        d = d or {}
        raw_versions = d.get("installed_versions")
        versions = (
            {str(k): str(v) for k, v in raw_versions.items()} if isinstance(raw_versions, dict) else {}
        )
        raw_artifacts = d.get("built_artifacts")
        artifacts = tuple(str(x) for x in raw_artifacts) if isinstance(raw_artifacts, (list, tuple)) else ()
        return cls(
            ok=bool(d.get("ok")),
            attempt_root=str(d.get("attempt_root") or ""),
            runtime=FrameworkRuntime.from_state(d.get("runtime")),
            built_artifacts=artifacts,
            installed_versions=versions,
            build_log_path=str(d.get("build_log_path") or ""),
            verify_log_path=str(d.get("verify_log_path") or ""),
            error=str(d.get("error") or ""),
            failure_class=normalize_failure_class(d.get("failure_class")),
            failure_summary=str(d.get("failure_summary") or ""),
        )


def build_novelty_key(
    action: TargetedBuildAction,
) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    """Repeat-vs-novel identity for the stall gate.

    Build requests are distinct across repositories and capabilities.
    """
    repo_url = action.repo_url.strip().rstrip("/").removesuffix(".git").lower()
    return (
        action.component,
        repo_url,
        action.capability,
        action.ref,
        action.gpu_arch,
        tuple(action.build_command),
    )


_GITHUB_PR_RE = _re.compile(r"https?://github\.com/([^/]+/[^/]+)/pull/(\d+)", _re.IGNORECASE)
_PR_REF_RE = _re.compile(r"^PR:(\d+)$")
# An issue is a discussion thread, not a branch: GitHub publishes
# ``refs/pull/{n}/head`` for pull requests but nothing checkoutable for issues.
_GITHUB_ISSUE_RE = _re.compile(r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)", _re.IGNORECASE)
_ISSUE_REF_RE = _re.compile(r"^issues?:(\d+)$", _re.IGNORECASE)


def resolve_build_ref(candidate: str, default_repo_url: str) -> tuple[str, str, str]:
    """Resolve a discovered candidate string to ``(repo_url, ref, source_pr_url)``.

    Handles the forms discovery and the enablement specialist produce:
    - ``https://github.com/{owner}/{repo}/pull/{n}`` → PR ref with full provenance.
    - ``PR:{n}`` bare ref → PR ref against ``default_repo_url``.
    - an issue URL or ``issue:{n}`` → repo with an EMPTY ref, i.e. fall back to
      tag autoselect (see below).
    - plain tag/branch/sha → verbatim ref against ``default_repo_url``.
    - any other URL → ``("", "", "")`` (skip; cannot derive a checkoutable ref).

    Issues need their own branch because they are *not* checkoutable. GitHub
    publishes ``refs/pull/{n}/head`` for a PR but exposes no ref for an issue,
    so an ``issue:{n}`` string that reaches ``git worktree add`` verbatim dies
    with ``fatal: invalid reference``. A specialist citing an upstream issue as
    the rationale for a from-source build is a normal and useful signal, so the
    issue number is dropped from the ref rather than the whole request being
    rejected: the empty ref makes the builder autoselect the newest matching
    tag, which is exactly where an issue fix would have landed. The issue URL is
    still returned as provenance so the audit trail keeps the citation.

    Args:
        candidate: The candidate ref string from discovery.
        default_repo_url: Repo URL to use when the candidate carries no origin.

    Returns:
        ``(repo_url, ref, source_pr_url)`` — all empty strings on skip.
    """
    s = candidate.strip()
    if not s:
        return ("", "", "")

    m = _GITHUB_PR_RE.match(s)
    if m:
        slug, number = m.group(1), m.group(2)
        repo_url = f"https://github.com/{slug}"
        return (repo_url, f"PR:{number}", s)

    if _PR_REF_RE.match(s):
        number = s.split(":", 1)[1]
        return (default_repo_url, f"PR:{number}", "")

    m = _GITHUB_ISSUE_RE.match(s)
    if m:
        slug = m.group(1)
        return (f"https://github.com/{slug}", "", s)

    if _ISSUE_REF_RE.match(s):
        return (default_repo_url, "", "")

    if "://" in s or s.startswith("git@"):
        return ("", "", "")

    return (default_repo_url, s, "")


__all__ = [
    "FAILURE_CLASSES",
    "BuildResult",
    "FrameworkRuntime",
    "TargetedBuildAction",
    "build_novelty_key",
    "normalize_failure_class",
    "resolve_build_ref",
]
