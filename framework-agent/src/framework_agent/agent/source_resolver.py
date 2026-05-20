"""Resolve vllm / sglang source roots for AST scanning.

Order of probes (first hit wins per framework):

1. Explicit env override: ``VLLM_SOURCE_ROOT`` / ``SGLANG_SOURCE_ROOT``.
2. Hyperloom container convention: ``/sgl-workspace/{vllm,sglang}/``.
3. ``importlib.util.find_spec`` → site-packages parent.

When none resolves, :class:`FrameworkSourceMissing` is raised so the
calling handler can short-circuit to
``OptimizeFailure(reason="source_not_found")``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Literal


# Path-relative-to-root substrings to skip during AST scan. Non-source
# noise plus test/examples directories that produce churn for zero
# useful flags. Mirrors the design §13 P2.2 whitelist intent. Matched
# against the *relative-to-source-root* path so we don't falsely
# exclude e.g. a fixture stored under a `tests/` directory of the
# scanner's own test suite.
ARG_SCAN_EXCLUDE: tuple[str, ...] = (
    "/tests/", "_test.py", "/test_",
    "/examples/", "/docs/", "/benchmarks/",
    "__pycache__/",
)

# Per-framework whitelist of directories / files most likely to expose
# tunable flags. Keeps the scan tight enough to finish in 1-3s single-
# threaded on a typical vllm/sglang tree.
ARG_SCAN_INCLUDE_DIRS: dict[str, tuple[str, ...]] = {
    "vllm": (
        "vllm/engine", "vllm/config.py", "vllm/entrypoints",
        "vllm/worker", "vllm/core/scheduler.py", "vllm/lora",
        "vllm/multimodal",
    ),
    "sglang": (
        "python/sglang/srt/server_args.py",
        "python/sglang/srt/managers",
        "python/sglang/srt/configs",
        "python/sglang/srt/layers",
    ),
}


class FrameworkSourceMissing(FileNotFoundError):
    """Raised when the requested framework's source root cannot be located."""


def _probe_one(framework: Literal["vllm", "sglang"]) -> Path | None:
    """Return the resolved source root for ``framework`` or None."""
    env_name = f"{framework.upper()}_SOURCE_ROOT"
    override = (os.environ.get(env_name) or "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            return p.resolve()

    container = Path(f"/sgl-workspace/{framework}")
    if container.is_dir():
        return container.resolve()

    spec = importlib.util.find_spec(framework)
    if spec is not None and spec.origin:
        origin = Path(spec.origin)
        parent = origin.parent if origin.name == "__init__.py" else origin.parent
        # site-packages layout: parent is e.g. .../site-packages/vllm
        # We want the package parent so paths like .../site-packages/vllm/engine
        # resolve correctly.
        candidate = parent.parent
        if candidate.is_dir() and (candidate / framework).is_dir():
            return candidate.resolve()
    return None


def resolve_framework_sources(
    frameworks: tuple[str, ...] = ("vllm", "sglang"),
) -> dict[str, Path]:
    """Return {framework -> resolved Path} for whichever are reachable.

    Frameworks that cannot be resolved are simply omitted from the
    return dict. Callers decide whether an empty dict is fatal (e.g.
    handler converts to OptimizeFailure(reason="source_not_found")).
    """
    out: dict[str, Path] = {}
    for fw in frameworks:
        fw_lower = fw.strip().lower()
        if fw_lower not in ARG_SCAN_INCLUDE_DIRS:
            continue
        p = _probe_one(fw_lower)  # type: ignore[arg-type]
        if p is not None:
            out[fw_lower] = p
    return out


def collect_target_files(framework: str, root: Path) -> list[Path]:
    """Walk the framework root and return whitelisted .py files.

    Deterministic ordering (sorted absolute paths) for reproducible
    scans + stable test fixtures.
    """
    fw_lower = framework.strip().lower()
    includes = ARG_SCAN_INCLUDE_DIRS.get(fw_lower)
    if includes is None:
        return []
    files: set[Path] = set()
    root_resolved = root.resolve()
    for sub in includes:
        p = root / sub
        if p.is_file() and p.suffix == ".py":
            files.add(p.resolve())
        elif p.is_dir():
            for f in p.rglob("*.py"):
                # Exclude check uses the path RELATIVE to ``root`` so a
                # fixture stored under e.g. tests/agent/fixtures/<root>/
                # is not falsely matched by the "/tests/" pattern.
                try:
                    rel = "/" + str(f.resolve().relative_to(root_resolved))
                except ValueError:
                    rel = str(f)
                if not any(x in rel for x in ARG_SCAN_EXCLUDE):
                    files.add(f.resolve())
    return sorted(files)


__all__ = [
    "ARG_SCAN_EXCLUDE",
    "ARG_SCAN_INCLUDE_DIRS",
    "FrameworkSourceMissing",
    "collect_target_files",
    "resolve_framework_sources",
]
