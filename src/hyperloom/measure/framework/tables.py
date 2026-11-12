# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tuned table each ``AITER_CONFIG_*`` variable of the installed aiter names, read from aiter's own source.

aiter's ``jit/core.py`` declares each variable twice: a module constant ``AITER_CONFIG_X = os.getenv("AITER_CONFIG_X",
f"{AITER_ROOT_DIR}/aiter/configs/<default>.csv")`` and a property of ``AITER_CONFIG`` that calls
``self.get_config_file("AITER_CONFIG_X", AITER_CONFIG_X, "<tuned>")``. The default file is what the variable names when
unset, and ``<tuned>`` is the pattern of the per-model tables aiter merges into it. The file is parsed, never imported:
importing aiter loads the GPU stack, and a parse that finds nothing just means no variable has a default to redirect.

``table_key`` is the one place that says which columns identify a row when aiter merges the tables a variable lists; the
tuned-table rules of ``hyperloom.measure`` (rows laid over a default table) read the key from it and nowhere else.
"""

from __future__ import annotations

import ast
import csv
import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AITER_VARIABLE_PREFIX", "TableVariable", "read_variables", "table_key"]

log = logging.getLogger(__name__)

AITER_VARIABLE_PREFIX = "AITER_CONFIG_"
_CONFIGS = "/aiter/configs/"
_ROOT_MARK = "\0"


@dataclass(frozen=True)
class TableVariable:
    """One variable: the file below ``aiter/configs`` it names when unset, and the name its per-model tables carry."""

    default: str
    tuned: str


def _is_getenv(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "getenv"


def _constant_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _default_file(value: ast.AST) -> str | None:
    """The file below ``aiter/configs`` that the f-string default inside ``value`` names, if there is one."""
    for node in ast.walk(value):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = "".join(_constant_str(part) or _ROOT_MARK for part in node.values)
        if text.startswith(_ROOT_MARK + _CONFIGS):
            return text.removeprefix(_ROOT_MARK + _CONFIGS)
    return None


def _module_defaults(tree: ast.Module) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [
            name
            for call in ast.walk(node.value)
            if _is_getenv(call) and call.args
            for name in [_constant_str(call.args[0])]
            if name and name.startswith(AITER_VARIABLE_PREFIX)
        ]
        default = _default_file(node.value)
        if names and default:
            defaults[names[0]] = default
    return defaults


def _tuned_names(tree: ast.Module) -> dict[str, str]:
    tuned: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_config_file":
            args = [_constant_str(arg) for arg in node.args]
            if len(args) == 3 and args[0] and args[2]:
                tuned[args[0]] = args[2]
    return tuned


def read_variables(core: Path) -> dict[str, TableVariable]:
    """The variables ``core`` (aiter's ``jit/core.py``) declares; empty when it is absent or has another shape."""
    try:
        tree = ast.parse(core.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        log.warning("aiter's variables cannot be read from %s: %s", core, exc)
        return {}
    tuned = _tuned_names(tree)
    return {name: TableVariable(file, tuned[name]) for name, file in _module_defaults(tree).items() if name in tuned}


def table_key(configs: Path, tuned: str, columns: Collection[str]) -> tuple[str, ...] | None:
    """The columns aiter identifies a row of the variable's merged tables by; ``None`` when it checks none.

    This is what ``AITER_CONFIG.update_config_files`` (``jit/core.py``) does with the tables a variable lists: the key is
    the header of the table's untuned sibling (``tuned`` with its last ``tuned`` read as ``untuned``, in ``configs``, the
    install's ``aiter/configs``) plus ``cu_num``, ``gfx`` and ``_tag``, kept to the ``columns`` the merged tables have.
    Two rows with one key make aiter rewrite the files it was given and raise. With no untuned sibling aiter merges
    without checking, so there is no key. The sibling's header is read from the install's own file; only the three
    added columns are aiter's rule written down, and this is the one place that does.
    """
    untuned = "untuned".join(tuned.rsplit("tuned", 1))
    try:
        with (configs / f"{untuned}.csv").open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, ValueError, csv.Error):
        return None
    wanted = [*header, *(name for name in ("cu_num", "gfx", "_tag") if name not in header)]
    return tuple(dict.fromkeys(name for name in wanted if name in columns))
