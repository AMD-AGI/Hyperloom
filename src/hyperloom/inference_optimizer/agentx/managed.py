# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind Magpie-managed serving to its immutable InferenceX client sources."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import canonical_sha256, native_workload_fingerprint


def project_root(path: str | Path) -> Path:
    """Accept a repository root or its packaged InferenceX project."""
    root = Path(path).expanduser().resolve()
    roots = [
        candidate
        for candidate in (root, root / "inferencex-e2e")
        if (candidate / "benchmarks/benchmark_lib.sh").is_file()
    ]
    if len(roots) > 1:
        raise ValueError(f"Ambiguous InferenceX project roots below {root}")
    return roots[0] if roots else root


def server_spec(benchmark: Mapping[str, Any]) -> dict[str, Any] | None:
    agentx = benchmark.get("agentx")
    resolved = agentx.get("resolved") if isinstance(agentx, Mapping) else None
    spec = resolved.get("server-launch-spec") if isinstance(resolved, Mapping) else None
    return spec if isinstance(spec, dict) else None


def git_text(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError(f"Cannot verify InferenceX source at {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def head_blob(root: Path, relative: str) -> bytes:
    prefix = git_text(root, "rev-parse", "--show-prefix")
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{prefix}{relative}"], capture_output=True, timeout=30, check=False
    )
    if result.returncode:
        raise ValueError(f"native AgentX execution input is not tracked at HEAD: {relative}")
    return result.stdout


def aiperf_revision(root: Path) -> str:
    """Verify the harness gitlink inside either supported project layout."""
    fields = git_text(root, "ls-tree", "HEAD", "utils/aiperf").split()
    if len(fields) != 4 or fields[0] != "160000" or not re.fullmatch(r"[0-9a-f]{40}", fields[2]):
        raise ValueError("InferenceX utils/aiperf is not a pinned git submodule")
    dependency = root / "utils/aiperf"
    if not (dependency / "pyproject.toml").is_file():
        raise ValueError("InferenceX utils/aiperf submodule is not initialized")
    if Path(git_text(dependency, "rev-parse", "--show-toplevel")).resolve() != dependency.resolve():
        raise ValueError("InferenceX utils/aiperf is not its own checkout")
    if git_text(dependency, "rev-parse", "HEAD") != fields[2]:
        raise ValueError("InferenceX utils/aiperf does not match its pinned gitlink")
    if git_text(dependency, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("InferenceX utils/aiperf submodule is not clean")
    return fields[2]


def _source_identity(root: Path, spec: Mapping[str, Any], config_file: str) -> dict[str, str]:
    sources = spec.get("source_files")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Magpie server launch specification has no recipe source files")
    relative_sources: dict[str, str] = {}
    for filename, expected in sources.items():
        path = Path(filename)
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"AgentX recipe source escapes InferenceX: {path}") from exc
        if not path.is_absolute() or any((root / parent).is_symlink() for parent in (relative, *relative.parents)):
            raise ValueError(f"AgentX recipe source must be an absolute path without symlinks: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected or path.read_bytes() != head_blob(root, relative.as_posix()):
            raise ValueError(f"AgentX recipe source differs from pinned HEAD: {relative}")
        relative_sources[relative.as_posix()] = actual
    required = {"benchmarks/srt_agentic.sh", "benchmarks/benchmark_lib.sh"}
    if config_file != "magpie:custom":
        required.update((config_file, "configs/runners.yaml"))
    if not required <= relative_sources.keys():
        raise ValueError("Magpie server launch source identity omits required client/recipe files")
    if git_text(root, "status", "--porcelain=v1", "--untracked-files=all", "--", "infx", *relative_sources):
        raise ValueError("native AgentX requires clean pinned client/recipe inputs")
    return relative_sources


def execution_identity(
    *,
    root: Path,
    benchmark: Mapping[str, Any],
    config_file: str,
    expected_ref: str,
    magpie: Mapping[str, Any],
    expected_magpie_ref: str,
    launch_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the managed contract without consulting obsolete shell manifests."""
    spec = server_spec(benchmark)
    if spec is None or spec.get("version") != 1 or benchmark.get("benchmark_script") != "srt_agentic.sh":
        raise ValueError("Managed AgentX requires server-launch-spec v1 and srt_agentic.sh")
    head = git_text(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", head) or spec.get("client_revision") != head:
        raise ValueError("Magpie server launch client revision does not match InferenceX HEAD")
    if expected_ref and expected_ref != head:
        raise ValueError(f"InferenceX HEAD {head} does not match the pinned ref {expected_ref}")
    magpie_commit = str(magpie.get("source_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", magpie_commit) or not re.fullmatch(
        r"[0-9a-f]{64}", str(magpie.get("fingerprint") or "")
    ):
        raise ValueError("Magpie resolver did not provide a valid execution identity")
    if expected_magpie_ref and magpie_commit != expected_magpie_ref:
        raise ValueError("Magpie source commit does not match the pinned ref")
    sources = _source_identity(root, spec, config_file)
    aiperf = aiperf_revision(root)
    if spec.get("aiperf_revision") != aiperf:
        raise ValueError("Magpie AIPerf revision does not match the client gitlink")
    identity = {
        "execution_owner": "magpie",
        "inferencex_commit": head,
        "launcher": "benchmarks/srt_agentic.sh",
        "config_file": config_file,
        "launcher_sha256": sources["benchmarks/srt_agentic.sh"],
        "aiperf_commit": aiperf,
        "magpie_commit": magpie_commit,
        "magpie_fingerprint": magpie["fingerprint"],
        "server_spec_sha256": canonical_sha256(spec),
        "transitive_inputs": sources,
    }
    identity["static_execution_fingerprint"] = canonical_sha256(identity)
    identity["workload_fingerprint"] = native_workload_fingerprint(benchmark, identity["static_execution_fingerprint"])
    identity["launch_config_sha256"] = canonical_sha256(
        {key: value for key, value in benchmark.items() if key != "workload_spec"}
    )
    identity["launch_environment"] = dict(launch_environment)
    identity["execution_fingerprint"] = canonical_sha256(identity)
    return identity
