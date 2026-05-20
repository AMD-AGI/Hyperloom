"""Per-file grep fallback for AST scanning.

Activated when ``libcst.parse_module`` raises ``ParserSyntaxError``
(file has unsupported syntax / non-UTF-8 / experimental walrus shapes).
Heuristic only — produces ``via='argparse_grep'`` or ``'dataclass_grep'``
markers so downstream KB writes (PR-G) can downgrade confidence per
design §9.3.
"""

from __future__ import annotations

import re
from pathlib import Path

from .flag_discovery import DiscoveredFlag, path_to_module


# argparse: parser.add_argument("--flag-name", ...)
# We deliberately allow optional spaces around ( and the leading "--"
# so multi-line add_argument calls still hit (the regex anchors on
# add_argument( + first quoted "--..." literal).
_ARGPARSE_RE = re.compile(
    r"add_argument\(\s*[\"'](--[a-z0-9][a-z0-9\-]*)[\"']"
    r"(?:[^)]*?type\s*=\s*(\w+))?"
    r"(?:[^)]*?default\s*=\s*([^,)]+?))?",
    re.DOTALL,
)

# dataclass / pydantic field: e.g. ``max_num_seqs: int = 256`` or
# ``flag: Optional[str] = None``. We only fire this inside a class body
# we've already classified as @dataclass / BaseModel (see the in_dc
# state machine in ``scan_module_via_grep``).
_FIELD_RE = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s*:\s*"
    r"([A-Za-z_][\w\[\], \.|]*)\s*=\s*(.+?)\s*$",
)

# Decorator + base markers used to flip the "we're inside a dataclass /
# BaseModel" state on / off. Bases group is optional (a plain
# ``class Foo:`` is valid Python and common for @dataclass classes).
_CLASS_DEC_DC_RE = re.compile(r"^\s*@(?:dataclass|dataclasses\.dataclass)")
_CLASS_DEF_RE = re.compile(r"^\s*class\s+(\w+)\s*(?:\((.*?)\))?\s*:")


def scan_module_via_grep(
    path: Path,
    *,
    framework: str = "",
    source_root: Path | None = None,
) -> list[DiscoveredFlag]:
    """Heuristic flag discovery via regex matching.

    Per-file invocation; the orchestrator (``ast_scanner.scan_framework_args``)
    decides when to fall back. Returns a list of :class:`DiscoveredFlag`
    objects; deduping is the caller's responsibility.
    """
    flags: list[DiscoveredFlag] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    module = (
        path_to_module(path, source_root) if source_root else path.stem
    )

    in_dc = False  # flips True when @dataclass deco seen on the next class
    next_class_is_dc = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _CLASS_DEC_DC_RE.match(line):
            next_class_is_dc = True
            continue
        m_class = _CLASS_DEF_RE.match(line)
        if m_class:
            bases = m_class.group(2) or ""
            in_dc = next_class_is_dc or ("BaseModel" in bases)
            next_class_is_dc = False
            continue
        if m := _ARGPARSE_RE.search(line):
            flag_name, type_hint, default = m.group(1), m.group(2), m.group(3)
            flags.append(DiscoveredFlag(
                flag_name=flag_name,
                module=module,
                source_path=str(path),
                line=lineno,
                via="argparse_grep",
                type_hint=(type_hint or "str").strip(),
                default_repr=(default or "_MISSING_").strip()[:120],
                help_text="",
                surface="cli",
                framework=framework,
            ))
        elif in_dc and (m_field := _FIELD_RE.match(line)):
            name, type_hint, default = m_field.group(1), m_field.group(2), m_field.group(3)
            # Filter out trivial fields likely not tunables (e.g.
            # `__slots__: tuple = ...`, double-underscore dunders).
            if name.startswith("_"):
                continue
            flags.append(DiscoveredFlag(
                flag_name=name,
                module=module,
                source_path=str(path),
                line=lineno,
                via="dataclass_grep",
                type_hint=type_hint.strip(),
                default_repr=default.strip()[:120],
                help_text="",
                surface="config",
                framework=framework,
            ))
    return flags


__all__ = ["scan_module_via_grep"]
