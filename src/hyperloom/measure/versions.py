"""Content versions of the socket implementations the product owns.

A version is the first 12 hex digits of a sha256 over the relative path and the
bytes of every source file of the implementation, so equal content gives equal
versions across sessions and any edit or rename changes it.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from importlib.abc import Traversable

__all__ = ["socket_versions"]

_VERSION_LENGTH = 12
_EXCLUDED_DIRS = frozenset({"tests", "__pycache__"})

# Package-relative sources behind each socket name. A directory entry is walked.
_MEASUREMENT_PARTS = ("measure", "sandbox")
_AI_WORK_FILES = (
    "orchestrator/specialists/agent_runner.py",
    "orchestrator/specialists/resources.py",
)


def _walk(node: Traversable, prefix: str) -> Iterable[tuple[str, Traversable]]:
    """Yield (relative path, file) for every .py file below node, skipping tests."""
    for child in node.iterdir():
        path = f"{prefix}/{child.name}"
        if child.is_dir():
            if child.name not in _EXCLUDED_DIRS:
                yield from _walk(child, path)
        elif child.name.endswith(".py"):
            yield path, child


def _digest(entries: Iterable[tuple[str, Traversable]]) -> str:
    """Hash path + bytes of every entry in sorted path order."""
    sha = hashlib.sha256()
    for path, node in sorted(entries, key=lambda item: item[0]):
        sha.update(path.encode("utf-8") + b"\0")
        data = node.read_bytes()
        sha.update(str(len(data)).encode("ascii") + b"\0" + data)
    return sha.hexdigest()[:_VERSION_LENGTH]


def _package_root() -> Traversable:
    return resources.files("hyperloom")


def socket_versions() -> dict[str, str]:
    """Return {socket name: 12 hex digits} for what this product owns."""
    root = _package_root()
    measurement: list[tuple[str, Traversable]] = []
    for part in _MEASUREMENT_PARTS:
        measurement.extend(_walk(root.joinpath(part), part))
    ai_work = []
    for rel in _AI_WORK_FILES:
        node = root
        for segment in rel.split("/"):
            node = node.joinpath(segment)
        ai_work.append((rel, node))
    return {"measurement": _digest(measurement), "ai_work": _digest(ai_work)}
