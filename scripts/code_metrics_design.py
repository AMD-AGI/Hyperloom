# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Diff-only design checks that ``scripts/code_metrics.py`` runs next to the ratchet.

Each check judges only what the change adds, in the production modules it touches (test
code is exempt from all of them). A line is *added* when the diff adds it and the change
did not remove the same text anywhere (a move, a re-indent, a renamed file). A function
or class is *new* when its ``def``/``class`` line is added and the same file at the base
(for a renamed file, the file it came from) had no unit of that qualified name. The rule
ids are those of the review rules' A family (code shape).

* **A1 private-name reach**: an added import, or attribute access through an imported
  name, of a private name (``_x``, not a dunder) or a module under a private package,
  owned by a different package unit. The units are the root packages and every module a
  contract in ``.importlinter`` names as a layer or an independent module; a module
  belongs to the longest unit that prefixes it.
* **A8 flag argument**: a new function whose first statement is ``if <flag>:`` (or
  ``if not <flag>:``, or a comparison of a two-value ``Literal`` parameter) and whose
  branch ends in ``return`` or ``raise``, where the flag is a ``bool`` or two-value
  ``Literal`` parameter: the flag picks one of two bodies. ``main``, click/typer
  commands and functions taking an ``argparse.Namespace`` are exempt.
* **A9 tuple returns**: a new function (not a dunder: ``__reduce__`` and kin have their
  shape fixed by the protocol) annotated to return ``tuple[...]`` (``typing.Tuple``, and
  inside ``Optional``/``Union``/``|``) of more than ``tuple-return-max-elements``
  elements, or returning a tuple display that long.
* **A10 raw vocabulary**: an added ``==``/``!=`` against, ``in`` over a display of, or
  ``case`` pattern of, a string literal that is the value of a member of exactly one ``Enum`` defined under
  ``vocabulary-roots``, in a module that names that enum (imports or defines it), when
  the other side is a member: its ``.value``, or a name for the enum's vocabulary
  (``status`` for ``RunStatus``). A value two enums share, one compared in a module that
  never names its enum, or one compared with raw input (``raw == "running"``: that input
  is being converted to the member) is another vocabulary or the conversion itself.
* **A11 hidden global write**: an added ``global`` statement, or store into ``globals()``.
* **A12 forwarder**: a new function (not a dunder) whose body is one ``return`` of a call
  passing exactly its own parameters through unchanged. A call of a builtin (by its bare
  name), of a class (a capitalised name: a factory), of ``.get`` (a table lookup) or of a
  method of a constant (``", ".join``) is a conversion or an accessor, not a forwarder;
  a decorated function (other than ``staticmethod``/``classmethod``: the decorator adds
  caching, registration, wrapping) and a method overriding one its base class defines (in
  ``vocabulary-roots``) exist for more than their body.
* **A12 single implementation**: a new ABC or ``Protocol`` with exactly one implementation
  (a subclass, or for a protocol a class defining all its methods) under
  ``vocabulary-roots``, in the same package unit; an abstraction a lower unit declares
  for a higher one to implement is the dependency inversion the layers ask for.
"""

from __future__ import annotations

import ast
import builtins
import configparser
import dataclasses
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import code_metrics_checks as checks
from code_metrics_collect import ToolError, Units, list_files

Problem = checks.Problem

PRIVATE_REACH = "private-reach"
FLAG_ARGUMENT = "flag-argument"
TUPLE_RETURN = "tuple-return"
RAW_VOCABULARY = "raw-vocabulary"
GLOBAL_WRITE = "global-write"
FORWARDER = "forwarder"
SINGLE_IMPLEMENTATION = "single-implementation"
#: Checks the A rules rate advisory: listed in the report, never failing the gate.
ADVISORY = frozenset({TUPLE_RETURN, RAW_VOCABULARY, GLOBAL_WRITE, SINGLE_IMPLEMENTATION})

_CONTRACT = "importlinter:contract:"
_UNIT_KEYS = ("layers", "modules")
_MODULE_NAME = re.compile(r"[A-Za-z_][\w.]*")
_SRC = "src/"
_INIT = "__init__"
_RECEIVERS = frozenset({"self", "cls"})
_BOOL = "bool"
_LITERAL = "Literal"
_TUPLES = frozenset({"tuple", "Tuple"})
_UNIONS = frozenset({"Optional", "Union"})
_ABSTRACT_BASES = frozenset({"ABC", "Protocol"})
_PROTOCOL = "Protocol"
_ABSTRACT_META = "ABCMeta"
_ENUM_BASES = frozenset({"Flag", "IntFlag"})
_ENUM_SUFFIX = "Enum"
#: Decorators of click and typer commands, by their last name (``@app.command()``).
_CLI_DECORATORS = frozenset({"command", "group", "option", "argument", "callback"})
_CLI_ENTRY = "main"
_CLI_NAMESPACE = "Namespace"
#: The decorator that leaves a method without a receiver.
_NO_RECEIVER = "staticmethod"
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_EQUALITY = (ast.Eq, ast.NotEq)
_SELECTOR_OPS = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)
_MEMBERSHIP = (ast.In, ast.NotIn)
_DISPLAYS = (ast.Tuple, ast.List, ast.Set)
_LITERAL_VALUES = 2
_VALUE = "value"
_GETATTR = "getattr"
_CAMEL_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+")
#: Callees that make a one-line function an accessor or a conversion rather than a forwarder.
_BUILTINS = frozenset(dir(builtins))
_LOOKUP = "get"
_GLOBALS = "globals"
#: The only decorators that add nothing to a function's body (any other caches, registers, wraps...).
_PLAIN_METHODS = frozenset({"staticmethod", "classmethod"})


@dataclasses.dataclass(frozen=True)
class Scope:
    """What the checks read from ``[tool.hyperloom.code_metrics]``."""

    roots: tuple[str, ...]
    exclude: tuple[str, ...]
    #: Text of the import-linter contracts that define the package units.
    contracts: str
    vocabulary_roots: tuple[str, ...]
    max_tuple: int


@dataclasses.dataclass(frozen=True)
class Packages:
    """The package units of the import-linter contracts."""

    roots: frozenset[str]
    #: Longest first, so the first prefix found is the innermost unit.
    units: tuple[str, ...]

    @classmethod
    def from_contracts(cls, text: str) -> Packages:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except configparser.Error as exc:
            raise ToolError(f"cannot read the import-linter contracts: {exc}") from exc
        roots = frozenset(_module_names(parser.get("importlinter", "root_packages", fallback="")))
        units = set(roots)
        for section in parser.sections():
            if section.startswith(_CONTRACT):
                for key in _UNIT_KEYS:
                    units.update(_module_names(parser.get(section, key, fallback="")))
        return cls(roots, tuple(sorted(units, key=len, reverse=True)))

    def unit(self, module: str) -> str | None:
        """The unit owning ``module``, or None when it is not one of the root packages."""
        if module.split(".")[0] not in self.roots:
            return None
        return next((u for u in self.units if module == u or module.startswith(u + ".")), None)


def _module_names(value: str) -> list[str]:
    return [name for line in value.splitlines() for name in _MODULE_NAME.findall(line.split("#")[0])]


def module_of(path: str) -> str:
    """The dotted module a repository path imports as (``src/`` is the import root)."""
    parts = list(Path(path.removeprefix(_SRC)).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] == _INIT:
        parts.pop()
    return ".".join(parts)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def is_private(name: str) -> bool:
    return name.startswith("_") and name != "_" and not _is_dunder(name)


def _terminal(node: ast.AST) -> str:
    """The last name of a decorator, base or annotation: ``b`` for ``a.b``, ``a.b(...)`` and ``a.b[...]``."""
    while isinstance(node, (ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _annotation(node: ast.AST | None) -> ast.AST | None:
    """``node``, with a string annotation parsed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    return node


def _body(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """The statements of ``function`` after its docstring."""
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[1:]
    return body


def _own_nodes(function: ast.AST) -> Iterator[ast.AST]:
    """Every node of ``function``'s body outside nested functions, lambdas and classes."""
    stack = [function]
    while stack:
        node = stack.pop()
        children = [child for child in ast.iter_child_nodes(node) if not isinstance(child, _SCOPES)]
        yield from children
        stack.extend(children)


@dataclasses.dataclass(frozen=True)
class Unit:
    """A function or class with its qualified name; ``in_class`` when a class holds it directly."""

    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    in_class: bool


def walk_units(node: ast.AST, prefix: str = "", in_class: bool = False) -> Iterator[Unit]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
            name = prefix + child.name
            yield Unit(name, child, in_class)
            yield from walk_units(child, name + ".", isinstance(child, ast.ClassDef))
        else:
            yield from walk_units(child, prefix, in_class)


@dataclasses.dataclass
class Change:
    """One touched production module: its tree and what the change adds to it."""

    path: str
    tree: ast.Module
    lines: list[str]
    added: set[int]
    #: Stripped text of every line the change removes, in any file.
    removed: set[str]
    #: Qualified names of the units the file (or its rename source) had at the base.
    base_units: set[str]

    def fresh(self, line: int) -> bool:
        """Whether the change wrote ``line`` rather than moved it."""
        if line not in self.added:
            return False
        return line > len(self.lines) or self.lines[line - 1].strip() not in self.removed

    def new_units(self) -> Iterator[Unit]:
        for unit in walk_units(self.tree):
            if unit.name not in self.base_units and self.fresh(unit.node.lineno):
                yield unit

    def new_functions(self) -> Iterator[Unit]:
        return (unit for unit in self.new_units() if isinstance(unit.node, _FUNCTIONS))


# --- A1 private-name reach ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Imported:
    """One imported name: what it binds, what the binding refers to, and the full dotted path."""

    bound: str
    target: str
    full: str
    line: int


def imported_names(node: ast.AST, module: str, is_package: bool) -> list[Imported]:
    if isinstance(node, ast.Import):
        found = []
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            found.append(Imported(bound, alias.name if alias.asname else bound, alias.name, alias.lineno))
        return found
    if not isinstance(node, ast.ImportFrom):
        return []
    base = _absolute(node, module, is_package)
    return [
        Imported(alias.asname or alias.name, f"{base}.{alias.name}", f"{base}.{alias.name}", alias.lineno)
        for alias in node.names
        if alias.name != "*"
    ]


def _absolute(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if not node.level:
        return node.module or ""
    parts = module.split(".")
    if not is_package:
        parts = parts[:-1]
    parts = parts[: max(len(parts) - (node.level - 1), 0)]
    return ".".join([*parts, node.module] if node.module else parts)


def private_reach_problems(change: Change, packages: Packages) -> list[Problem]:
    module = module_of(change.path)
    own = packages.unit(module) or module.split(".")[0]
    bindings: dict[str, str] = {}
    found: dict[int, str] = {}
    for node in ast.walk(change.tree):
        for name in imported_names(node, module, change.path.endswith(f"{_INIT}.py")):
            bindings[name.bound] = name.target
            reach = _reach(name.full, packages, own) if change.fresh(name.line) else None
            if reach:
                found.setdefault(name.line, reach)
    for node, local in _attributes(change.tree, frozenset()):
        if is_private(node.attr) and change.fresh(node.lineno) and _root(node) not in local:
            owner = _resolve(node.value, bindings)
            reach = _reach(f"{owner}.{node.attr}", packages, own) if owner else None
            if reach:
                found.setdefault(node.lineno, reach)
    return [
        Problem(
            PRIVATE_REACH,
            change.path,
            line,
            f"A1 private-name reach: {reach} from `{own}`; make it public in its owner, or move the caller there",
        )
        for line, reach in sorted(found.items())
    ]


def _attributes(node: ast.AST, local: frozenset[str]) -> Iterator[tuple[ast.Attribute, frozenset[str]]]:
    """Every attribute access under ``node``, with the names the enclosing functions bind locally."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_FUNCTIONS, ast.Lambda)):
            yield from _attributes(child, local | _locals(child))
            continue
        if isinstance(child, ast.Attribute):
            yield child, local
        yield from _attributes(child, local)


def _locals(function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> frozenset[str]:
    """The parameters of ``function`` and the names its own body assigns (a ``global`` name is not local)."""
    args = function.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if a}
    declared = set()
    for node in _own_nodes(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
    return frozenset(names - declared)


def _root(node: ast.expr) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _reach(dotted: str, packages: Packages, own: str) -> str | None:
    """How ``dotted`` reaches into another unit's private name, or None when it does not."""
    unit = packages.unit(dotted)
    if unit is None or unit == own:
        return None
    private = next((part for part in dotted[len(unit) :].split(".") if is_private(part)), None)
    return None if private is None else f"`{dotted}` uses `{private}`, private to `{unit}`"


def _resolve(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve(node.value, bindings)
        return f"{owner}.{node.attr}" if owner else None
    return None


# --- A8 flag argument -------------------------------------------------------------


def flag_argument_problems(change: Change) -> list[Problem]:
    problems = []
    for unit in change.new_functions():
        flag = None if _is_cli(unit.node) else _selecting_flag(unit.node)
        if flag:
            problems.append(
                Problem(
                    FLAG_ARGUMENT,
                    change.path,
                    unit.node.lineno,
                    f"A8 flag argument: `{unit.name}` starts by branching on its flag `{flag}` into two bodies; "
                    "split it into one function per value and let the caller pick",
                )
            )
    return problems


def _is_cli(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if function.name == _CLI_ENTRY or any(_terminal(d) in _CLI_DECORATORS for d in function.decorator_list):
        return True
    return any(_terminal(_annotation(arg.annotation)) == _CLI_NAMESPACE for arg in _all_args(function))


def _all_args(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    args = function.args
    return [*args.posonlyargs, *args.args, *args.kwonlyargs]


def _selecting_flag(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = _body(function)
    if not body or not isinstance(body[0], ast.If) or not isinstance(body[0].body[-1], (ast.Return, ast.Raise)):
        return None
    tested = _tested_name(body[0].test)
    return tested if tested in _flag_parameters(function) else None


def _tested_name(test: ast.expr) -> str | None:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], _SELECTOR_OPS):
        left, right = test.left, test.comparators[0]
        test = left if isinstance(right, ast.Constant) else right if isinstance(left, ast.Constant) else test
    return test.id if isinstance(test, ast.Name) else None


def _flag_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = [*zip(positional, defaults, strict=True), *zip(args.kwonlyargs, args.kw_defaults, strict=True)]
    return {arg.arg for arg, default in pairs if _is_flag(arg, default)}


def _is_flag(arg: ast.arg, default: ast.expr | None) -> bool:
    if isinstance(default, ast.Constant) and isinstance(default.value, bool):
        return True
    annotation = _annotation(arg.annotation)
    if isinstance(annotation, ast.Name):
        return annotation.id == _BOOL
    if isinstance(annotation, ast.Subscript) and _terminal(annotation.value) == _LITERAL:
        values = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        return len(values) == _LITERAL_VALUES
    return False


# --- A9 tuple returns -------------------------------------------------------------


def tuple_return_problems(change: Change, max_elements: int) -> list[Problem]:
    problems = []
    for unit in change.new_functions():
        if _is_dunder(unit.node.name):
            continue
        returns = _annotation(unit.node.returns)
        # An annotation that admits any length (``tuple[X, ...]``) is what the displays return.
        returned = 0 if _is_variadic(returns) else _returned_size(unit.node)
        size = max(_annotated_size(returns), returned)
        if size > max_elements:
            problems.append(
                Problem(
                    TUPLE_RETURN,
                    change.path,
                    unit.node.lineno,
                    f"A9 tuple return: `{unit.name}` returns a {size}-element tuple (limit {max_elements}); "
                    "return a dataclass or NamedTuple whose fields name the parts",
                )
            )
    return problems


def _annotated_size(node: ast.AST | None) -> int:
    if isinstance(node, ast.BinOp):
        return max(_annotated_size(node.left), _annotated_size(node.right))
    if not isinstance(node, ast.Subscript):
        return 0
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if _terminal(node.value) in _UNIONS:
        return max((_annotated_size(e) for e in elements), default=0)
    if _terminal(node.value) not in _TUPLES:
        return 0
    if any(isinstance(e, ast.Constant) and e.value is Ellipsis for e in elements):
        return 0
    return len(elements)


def _is_variadic(node: ast.AST | None) -> bool:
    return node is not None and any(
        isinstance(sub, ast.Subscript)
        and _terminal(sub.value) in _TUPLES
        and isinstance(sub.slice, ast.Tuple)
        and any(isinstance(e, ast.Constant) and e.value is Ellipsis for e in sub.slice.elts)
        for sub in ast.walk(node)
    )


def _returned_size(function: ast.AST) -> int:
    sizes = [
        len(node.value.elts)
        for node in _own_nodes(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]
    return max(sizes, default=0)


# --- A10 raw vocabulary -----------------------------------------------------------


def raw_vocabulary_problems(change: Change, vocabulary: dict[str, list[str]]) -> list[Problem]:
    named = _names(change.tree)
    owned = {
        value: members[0]
        for value, members in vocabulary.items()
        if len(set(members)) == 1 and members[0].split(".")[0] in named
    }
    found: dict[int, tuple[ast.Constant, str]] = {}
    for test, subject, string in _comparisons(change.tree):
        member = owned.get(string.value)
        if member and change.fresh(string.lineno) and _holds(subject, member.split(".")[0]):
            found.setdefault(id(test), (string, member))
    return [
        Problem(
            RAW_VOCABULARY,
            change.path,
            string.lineno,
            f"A10 raw vocabulary: compares the string {string.value!r}, the value of `{member}`; "
            f"compare with `{member}` (convert the input to the enum once, where it enters)",
        )
        for string, member in sorted(found.values(), key=lambda hit: hit[0].lineno)
    ]


def _comparisons(tree: ast.Module) -> Iterator[tuple[ast.AST, ast.expr, ast.Constant]]:
    """(test, subject, string) of every ``==``/``!=``, ``in`` over a display and ``case`` with a string."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            yield from _case_strings(node)
        elif isinstance(node, ast.Compare):
            yield from _compare_strings(node)


def _case_strings(node: ast.Match) -> Iterator[tuple[ast.AST, ast.expr, ast.Constant]]:
    for case in node.cases:
        for pattern in ast.walk(case.pattern):
            if isinstance(pattern, ast.MatchValue) and _is_string(pattern.value):
                yield pattern, node.subject, pattern.value


def _compare_strings(node: ast.Compare) -> Iterator[tuple[ast.AST, ast.expr, ast.Constant]]:
    operands = [node.left, *node.comparators]
    for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
        if isinstance(op, _EQUALITY):
            yield from ((node, b, a) for a, b in ((left, right), (right, left)) if _is_string(a))
        elif isinstance(op, _MEMBERSHIP) and isinstance(right, _DISPLAYS):
            yield from ((node, left, e) for e in right.elts if _is_string(e))


def _is_string(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _holds(subject: ast.expr, enum: str) -> bool:
    """Whether ``subject`` reads as a member of ``enum``: its ``.value``, or a name for the enum's vocabulary.

    A name is for the vocabulary when its snake_case words and the enum's words end alike
    (``status`` or ``run_status`` for ``RunStatus``, ``failure_type`` for
    ``SpecialistFailureType``). Any
    other subject (``raw``, ``v``, ``str(x)``) is input on its way in, the one place a
    string is compared to build the member.
    """
    if isinstance(subject, ast.Attribute) and subject.attr == _VALUE:
        return True
    if isinstance(subject, ast.Call) and _terminal(subject.func) == _GETATTR and len(subject.args) > 1:
        return isinstance(subject.args[1], ast.Constant) and subject.args[1].value == _VALUE
    if isinstance(subject, ast.Name):
        name = subject.id
    elif isinstance(subject, ast.Attribute):
        name = subject.attr
    else:
        return False
    words = [w for w in name.lower().split("_") if w]
    vocabulary = [w.lower() for w in _CAMEL_WORDS.findall(enum)]
    common = min(len(words), len(vocabulary))
    return common > 0 and words[-common:] == vocabulary[-common:]


def _names(tree: ast.Module) -> set[str]:
    """Every name ``tree`` binds, reads or defines a class as, and the last part of each attribute."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Attribute, ast.ClassDef, ast.alias)):
            names.add(node.attr if isinstance(node, ast.Attribute) else node.name.split(".")[-1])
    return names


def enum_values(trees: Iterable[ast.Module]) -> dict[str, list[str]]:
    """String value -> the ``Enum.MEMBER`` of every enum class in ``trees`` that has it."""
    found: dict[str, list[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(_is_enum_base(base) for base in node.bases):
                for name, value in _members(node):
                    found.setdefault(value, []).append(f"{node.name}.{name}")
    return found


def _is_enum_base(base: ast.expr) -> bool:
    name = _terminal(base)
    return name.endswith(_ENUM_SUFFIX) or name in _ENUM_BASES


def _members(node: ast.ClassDef) -> Iterator[tuple[str, str]]:
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and not stmt.targets[0].id.startswith("_")
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
            and stmt.value.value
        ):
            yield stmt.targets[0].id, stmt.value.value


# --- A11 hidden global write ------------------------------------------------------


def global_write_problems(change: Change) -> list[Problem]:
    return [
        Problem(
            GLOBAL_WRITE,
            change.path,
            node.lineno,
            f"A11 hidden global write: `global {', '.join(node.names)}` rebinds module state from inside a "
            "function; pass the value in and return the result, or keep the state on an object the caller owns",
        )
        for node in ast.walk(change.tree)
        if isinstance(node, ast.Global) and change.fresh(node.lineno)
    ] + [
        Problem(
            GLOBAL_WRITE,
            change.path,
            node.lineno,
            "A11 hidden global write: `globals()[...]` rebinds module state; pass the value in and return the "
            "result, or keep the state on an object the caller owns",
        )
        for node in ast.walk(change.tree)
        if _is_globals_store(node) and change.fresh(node.lineno)
    ]


def _is_globals_store(node: ast.AST) -> bool:
    """``globals()[name] = ...`` (or ``del``, or an augmented assignment)."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == _GLOBALS
    )


# --- A12 forwarders and single implementations ------------------------------------


def forwarder_problems(change: Change, index: list[ClassInfo]) -> list[Problem]:
    problems = []
    overrides = _overrides(change.tree, index)
    for unit in change.new_functions():
        callee = None if id(unit.node) in overrides else _forwarded_to(unit)
        if callee:
            problems.append(
                Problem(
                    FORWARDER,
                    change.path,
                    unit.node.lineno,
                    f"A12 forwarder: `{unit.name}` only passes its parameters to `{callee}`; call `{callee}` "
                    "directly, or give this layer a job of its own",
                )
            )
    return problems


def _forwarded_to(unit: Unit) -> str | None:
    function = unit.node
    decorators = {_terminal(d) for d in function.decorator_list}
    if function.name.startswith("__") or decorators - _PLAIN_METHODS:
        return None
    body = _body(function)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    call = body[0].value.value if isinstance(body[0].value, ast.Await) else body[0].value
    receiver = unit.in_class and _NO_RECEIVER not in decorators
    params = _parameter_names(function, receiver)
    if not isinstance(call, ast.Call) or not params or not _forwards_to(call.func, params):
        return None
    passed = _passed_names(call)
    return ast.unparse(call.func) if passed is not None and sorted(passed) == sorted(params) else None


def _forwards_to(callee: ast.expr, params: list[str]) -> bool:
    """Whether calling ``callee`` is forwarding: not a builtin, a factory, a lookup or a method of a constant."""
    name = _terminal(callee)
    if isinstance(callee, ast.Name):
        return name not in params and name not in _BUILTINS and not name[:1].isupper()
    if not isinstance(callee, ast.Attribute) or isinstance(callee.value, ast.Constant):
        return False
    return name != _LOOKUP and not name[:1].isupper()


def _overrides(tree: ast.Module, index: list[ClassInfo]) -> set[int]:
    """The ``id`` of every method in ``tree`` that overrides a method a known base class defines."""
    defined: dict[str, set[str]] = {}
    for info in index:
        defined.setdefault(info.name, set()).update(info.methods)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            inherited = set().union(*(defined.get(_terminal(base), set()) for base in node.bases))
            found.update(id(stmt) for stmt in node.body if isinstance(stmt, _FUNCTIONS) and stmt.name in inherited)
    return found


def _parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef, receiver: bool) -> list[str]:
    args = function.args
    names = [a.arg for a in _all_args(function)]
    if receiver and names and names[0] in _RECEIVERS:
        names = names[1:]
    return names + [a.arg for a in (args.vararg, args.kwarg) if a is not None]


def _passed_names(call: ast.Call) -> list[str] | None:
    """The names ``call`` passes as they are, or None when it passes anything else."""
    passed = []
    for arg in call.args:
        value = arg.value if isinstance(arg, ast.Starred) else arg
        if not isinstance(value, ast.Name):
            return None
        passed.append(value.id)
    for keyword in call.keywords:
        if not isinstance(keyword.value, ast.Name) or keyword.arg not in (None, keyword.value.id):
            return None
        passed.append(keyword.value.id)
    return passed


@dataclasses.dataclass(frozen=True)
class ClassInfo:
    """What the single-implementation check needs of one class."""

    path: str
    name: str
    bases: frozenset[str]
    methods: frozenset[str]
    is_protocol: bool


def class_index(trees: Iterable[tuple[str, ast.Module]]) -> list[ClassInfo]:
    return [
        ClassInfo(path, node.name, frozenset(_terminal(b) for b in node.bases), _methods(node), _is_protocol(node))
        for path, tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]


def _methods(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(stmt.name for stmt in node.body if isinstance(stmt, _FUNCTIONS))


def _is_protocol(node: ast.ClassDef) -> bool:
    return any(_terminal(base) == _PROTOCOL for base in node.bases)


def _is_abstract(node: ast.ClassDef) -> bool:
    if any(_terminal(base) in _ABSTRACT_BASES for base in node.bases):
        return True
    return any(_terminal(keyword.value) == _ABSTRACT_META for keyword in node.keywords)


def single_implementation_problems(change: Change, index: list[ClassInfo], packages: Packages) -> list[Problem]:
    problems = []
    own = packages.unit(module_of(change.path))
    for unit in change.new_units():
        node = unit.node
        if not isinstance(node, ast.ClassDef) or not _is_abstract(node):
            continue
        implementations = _implementations(node, index, change.path)
        if len(implementations) == 1 and packages.unit(module_of(implementations[0].path)) == own:
            problems.append(
                Problem(
                    SINGLE_IMPLEMENTATION,
                    change.path,
                    node.lineno,
                    f"A12 single implementation: `{unit.name}` is abstract and only "
                    f"`{implementations[0].name}` ({implementations[0].path}) implements it; use the implementation "
                    "until a second one exists",
                )
            )
    return problems


def _implementations(node: ast.ClassDef, index: list[ClassInfo], path: str) -> list[ClassInfo]:
    methods = _methods(node)
    structural = _is_protocol(node) and bool(methods)
    return [
        info
        for info in index
        if (info.path, info.name) != (path, node.name)
        and (node.name in info.bases or (structural and not info.is_protocol and methods <= info.methods))
    ]


# --- all of the above ---------------------------------------------------------------


def read_change(root: Path, since: str, path: str, diff: checks.Diff, units: Units) -> Change:
    text = (root / path).read_text(encoding=checks.ENCODING)
    base = checks.show(root, since, diff.renames.get(path, path))
    try:
        base_units = {unit.name for unit in walk_units(ast.parse(base, filename=path))} if base else set()
    except (SyntaxError, ValueError) as exc:
        raise ToolError(f"cannot parse {path} at the base: {exc}") from exc
    return Change(path, units.tree(path), text.splitlines(), diff.added.get(path, set()), diff.removed, base_units)


def design_problems(root: Path, scope: Scope, units: Units, since: str) -> list[Problem]:
    """Every design refusal for the change from ``since`` to the working tree."""
    diff = checks.read_diff(root, since)
    production = [p for p in list_files(root, scope.roots, scope.exclude)[0] if not checks.is_test_path(p)]
    changed = [p for p in production if diff.added.get(p)]
    if not changed:
        return []
    packages = Packages.from_contracts(scope.contracts)
    corpus = [(p, units.tree(p)) for p in production if _is_under(p, scope.vocabulary_roots)]
    vocabulary = enum_values(tree for _, tree in corpus)
    index = class_index(corpus)
    problems: list[Problem] = []
    for path in changed:
        change = read_change(root, since, path, diff, units)
        problems += private_reach_problems(change, packages)
        problems += flag_argument_problems(change)
        problems += tuple_return_problems(change, scope.max_tuple)
        problems += raw_vocabulary_problems(change, vocabulary)
        problems += global_write_problems(change)
        problems += forwarder_problems(change, index)
        problems += single_implementation_problems(change, index, packages)
    return problems


def _is_under(path: str, roots: Iterable[str]) -> bool:
    return any(path.startswith(root.rstrip("/") + "/") for root in roots)
