"""Compile and evaluate collection-mapping expressions.

An expression is a YAML/JSON node:

* ``"$name.key.0.key"`` reads a path from a bound name; a missing step is ``None``.
* A string containing ``{$path}`` placeholders is a template.
* ``"$$text"`` is the literal ``"$text"``; any other scalar is itself.
* A list evaluates each item.
* A mapping is one built-in call, named by exactly one head key.

Mappings compile once, so an unknown built-in, a malformed argument, or a path
root that is not bound at that point fails when the mapping is loaded rather
than on the first document that reaches it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

Scope = Mapping[str, Any]

_PATH_RE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)((?:\.[A-Za-z0-9_-]+)*)$")
_PLACEHOLDER_RE = re.compile(r"\{(\$[^{}]+)\}")
_TOKEN_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.:@/-]+")
_TOKEN_MAX_CHARS = 200


class MappingError(ValueError):
    """Raised when a collection mapping is malformed."""


class EvaluationError(ValueError):
    """Raised when one unit's data cannot produce a value the mapping requires."""


class Expression(Protocol):
    def evaluate(self, scope: Scope) -> Any:
        """The value this expression takes in ``scope``."""


def is_present(value: Any) -> bool:
    """Whether a value carries information: not ``None``, ``""``, ``[]`` or ``{}``."""

    return value is not None and value != "" and value != [] and value != {}


def truth(value: Any) -> bool:
    """Condition semantics: booleans are themselves, anything else is presence."""

    return value if isinstance(value, bool) else is_present(value)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"value is not canonical JSON: {exc}") from exc


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return canonical_json(value)


@dataclass(frozen=True)
class Literal:
    value: Any

    def evaluate(self, scope: Scope) -> Any:
        return self.value


@dataclass(frozen=True)
class Path:
    root: str
    steps: tuple[str, ...]

    def evaluate(self, scope: Scope) -> Any:
        current = scope.get(self.root)
        for step in self.steps:
            if isinstance(current, Mapping):
                current = current.get(step)
            elif isinstance(current, list) and step.isdigit():
                index = int(step)
                current = current[index] if index < len(current) else None
            else:
                return None
        return current


@dataclass(frozen=True)
class Template:
    parts: tuple[str | Path, ...]

    def evaluate(self, scope: Scope) -> str:
        return "".join(part if isinstance(part, str) else _format(part.evaluate(scope)) for part in self.parts)


@dataclass(frozen=True)
class ListOf:
    items: tuple[Expression, ...]

    def evaluate(self, scope: Scope) -> list[Any]:
        return [item.evaluate(scope) for item in self.items]


@dataclass(frozen=True)
class Unary:
    name: str
    function: Callable[[Any], Any]
    argument: Expression

    def evaluate(self, scope: Scope) -> Any:
        return self.function(self.argument.evaluate(scope))


@dataclass(frozen=True)
class Variadic:
    name: str
    function: Callable[[Sequence[Expression], Scope], Any]
    arguments: tuple[Expression, ...]

    def evaluate(self, scope: Scope) -> Any:
        return self.function(self.arguments, scope)


@dataclass(frozen=True)
class ObjectOf:
    fields: tuple[tuple[str, Expression], ...]

    def evaluate(self, scope: Scope) -> dict[str, Any]:
        return {name: expression.evaluate(scope) for name, expression in self.fields}


@dataclass(frozen=True)
class IfThen:
    condition: Expression
    then: Expression
    otherwise: Expression

    def evaluate(self, scope: Scope) -> Any:
        branch = self.then if truth(self.condition.evaluate(scope)) else self.otherwise
        return branch.evaluate(scope)


@dataclass(frozen=True)
class MapTable:
    key: Expression
    table: tuple[tuple[str, Any], ...]
    default: Expression

    def evaluate(self, scope: Scope) -> Any:
        key = self.key.evaluate(scope)
        if key is not None:
            for candidate, value in self.table:
                if candidate == str(key):
                    return value
        return self.default.evaluate(scope)


@dataclass(frozen=True)
class Each:
    source: Expression
    name: str
    where: Expression | None
    value: Expression | None

    def evaluate(self, scope: Scope) -> list[Any]:
        items = self.source.evaluate(scope)
        if items is None:
            return []
        if not isinstance(items, list):
            raise EvaluationError(f"each expects a list, got {type(items).__name__}")
        out: list[Any] = []
        for item in items:
            local = {**scope, self.name: item}
            if self.where is not None and not truth(self.where.evaluate(local)):
                continue
            out.append(item if self.value is None else self.value.evaluate(local))
        return out


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _lower(value: Any) -> str | None:
    return None if value is None else _text(value).lower()


def _upper(value: Any) -> str | None:
    return None if value is None else _text(value).upper()


def _squash(value: Any) -> str:
    return " ".join(_text(value).split())


def _token(value: Any) -> str | None:
    normalized = _TOKEN_UNSAFE_RE.sub("_", _text(value)).strip("_")
    return normalized[:_TOKEN_MAX_CHARS] or None


def _string_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationError(f"string_map expects a mapping, got {type(value).__name__}")
    out: dict[str, str] = {}
    for key, item in value.items():
        name = _text(key)
        if not name:
            raise EvaluationError("string_map found an empty key")
        out[name] = str(item)
    return dict(sorted(out.items()))


def _sorted_set(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationError(f"sorted_set expects a list, got {type(value).__name__}")
    return sorted({_text(item) for item in value if _text(item)})


def _sha256(value: Any) -> str:
    data = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(data.encode()).hexdigest()


def _length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, list, Mapping)):
        return len(value)
    raise EvaluationError(f"length expects a string, list, or mapping, got {type(value).__name__}")


def _compact(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationError(f"compact expects a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items() if item is not None and item != ""}


def _first(arguments: Sequence[Expression], scope: Scope) -> Any:
    for argument in arguments:
        value = argument.evaluate(scope)
        if is_present(value):
            return value
    return None


def _concat(arguments: Sequence[Expression], scope: Scope) -> list[Any]:
    out: list[Any] = []
    for argument in arguments:
        value = argument.evaluate(scope)
        if value is None:
            continue
        if not isinstance(value, list):
            raise EvaluationError(f"concat expects lists, got {type(value).__name__}")
        out.extend(value)
    return out


def _hash48(arguments: Sequence[Expression], scope: Scope) -> int:
    basis = "\0".join(_format(argument.evaluate(scope)) for argument in arguments)
    return int.from_bytes(hashlib.sha256(basis.encode()).digest()[:6], "big")


def _all(arguments: Sequence[Expression], scope: Scope) -> bool:
    return all(truth(argument.evaluate(scope)) for argument in arguments)


def _any(arguments: Sequence[Expression], scope: Scope) -> bool:
    return any(truth(argument.evaluate(scope)) for argument in arguments)


def _pair(arguments: Sequence[Expression], scope: Scope) -> tuple[Any, Any]:
    return arguments[0].evaluate(scope), arguments[1].evaluate(scope)


def _eq(arguments: Sequence[Expression], scope: Scope) -> bool:
    left, right = _pair(arguments, scope)
    return bool(left == right)


def _ne(arguments: Sequence[Expression], scope: Scope) -> bool:
    return not _eq(arguments, scope)


def _gt(arguments: Sequence[Expression], scope: Scope) -> bool:
    left, right = _pair(arguments, scope)
    left_number, right_number = _number(left), _number(right)
    return left_number is not None and right_number is not None and left_number > right_number


def _members(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluationError(f"membership expects a list, got {type(value).__name__}")
    return value


def _in(arguments: Sequence[Expression], scope: Scope) -> bool:
    value, members = _pair(arguments, scope)
    return value in _members(members)


def _not_in(arguments: Sequence[Expression], scope: Scope) -> bool:
    return not _in(arguments, scope)


def _min_length(arguments: Sequence[Expression], scope: Scope) -> bool:
    value, minimum = _pair(arguments, scope)
    bound = _number(minimum)
    return isinstance(value, str) and bound is not None and len(value) >= bound


def _starts_with_any(arguments: Sequence[Expression], scope: Scope) -> bool:
    value, prefixes = _pair(arguments, scope)
    return isinstance(value, str) and any(
        isinstance(prefix, str) and value.startswith(prefix) for prefix in _members(prefixes)
    )


def _rstrip(arguments: Sequence[Expression], scope: Scope) -> str | None:
    value, characters = _pair(arguments, scope)
    return None if value is None else str(value).rstrip(_text(characters))


UNARY: dict[str, Callable[[Any], Any]] = {
    "text": _text,
    "lower": _lower,
    "upper": _upper,
    "squash": _squash,
    "token": _token,
    "number": _number,
    "integer": _integer,
    "string_map": _string_map,
    "sorted_set": _sorted_set,
    "canonical_json": canonical_json,
    "sha256": _sha256,
    "length": _length,
    "compact": _compact,
    "present": is_present,
    "absent": lambda value: not is_present(value),
    "is_bool": lambda value: isinstance(value, bool),
    "not": lambda value: not truth(value),
}

VARIADIC: dict[str, tuple[Callable[[Sequence[Expression], Scope], Any], int | None]] = {
    "first": (_first, None),
    "concat": (_concat, None),
    "hash48": (_hash48, None),
    "all": (_all, None),
    "any": (_any, None),
    "eq": (_eq, 2),
    "ne": (_ne, 2),
    "gt": (_gt, 2),
    "in": (_in, 2),
    "not_in": (_not_in, 2),
    "min_length": (_min_length, 2),
    "starts_with_any": (_starts_with_any, 2),
    "rstrip": (_rstrip, 2),
}

STRUCTURED: dict[str, frozenset[str]] = {
    "object": frozenset({"object"}),
    "if": frozenset({"if", "then", "else"}),
    "map": frozenset({"map", "table", "default"}),
    "each": frozenset({"each", "as", "where", "value"}),
}

BUILTINS = frozenset(UNARY) | frozenset(VARIADIC) | frozenset(STRUCTURED)
_NONE = Literal(None)


def _compile_path(text: str, names: frozenset[str], where: str) -> Path | None:
    match = _PATH_RE.fullmatch(text)
    if match is None:
        return None
    root = match.group(1)
    if root not in names:
        raise MappingError(f"{where}: ${root} is not bound here (bound: {', '.join(sorted(names))})")
    steps = tuple(step for step in match.group(2).split(".") if step)
    return Path(root, steps)


def _compile_string(text: str, names: frozenset[str], where: str) -> Expression:
    if text.startswith("$$"):
        return Literal(text[1:])
    path = _compile_path(text, names, where)
    if path is not None:
        return path
    if text.startswith("$"):
        raise MappingError(f"{where}: {text!r} is not a valid path")
    if "{$" not in text:
        return Literal(text)
    parts: list[str | Path] = []
    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        placeholder = _compile_path(match.group(1), names, where)
        if placeholder is None:
            raise MappingError(f"{where}: template placeholder {match.group(0)!r} is not a path")
        if match.start() > cursor:
            parts.append(text[cursor : match.start()])
        parts.append(placeholder)
        cursor = match.end()
    if cursor < len(text):
        parts.append(text[cursor:])
    return Template(tuple(parts))


def _binding_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise MappingError(f"{where}: 'as' must be an identifier")
    return value


def _compile_call(node: Mapping[str, Any], names: frozenset[str], where: str) -> Expression:
    heads = [key for key in node if key in BUILTINS]
    if len(heads) != 1:
        raise MappingError(
            f"{where}: a mapping must be exactly one built-in call "
            f"(known: {', '.join(sorted(BUILTINS))}); got keys {sorted(map(str, node))}"
        )
    head = str(heads[0])
    allowed = STRUCTURED.get(head, frozenset({head}))
    extra = sorted(str(key) for key in node if key not in allowed)
    if extra:
        raise MappingError(f"{where}: {head} does not accept {', '.join(extra)}")
    location = f"{where}.{head}"
    argument = node[head]
    if head in UNARY:
        return Unary(head, UNARY[head], compile_expression(argument, names, location))
    if head in VARIADIC:
        function, arity = VARIADIC[head]
        if not isinstance(argument, list):
            raise MappingError(f"{location}: expects a list of arguments")
        if arity is not None and len(argument) != arity:
            raise MappingError(f"{location}: expects exactly {arity} arguments")
        return Variadic(
            head,
            function,
            tuple(compile_expression(item, names, f"{location}[{index}]") for index, item in enumerate(argument)),
        )
    if head == "object":
        if not isinstance(argument, Mapping):
            raise MappingError(f"{location}: expects a mapping of field expressions")
        return ObjectOf(
            tuple((str(key), compile_expression(value, names, f"{location}.{key}")) for key, value in argument.items())
        )
    if head == "if":
        if "then" not in node:
            raise MappingError(f"{where}: if requires then")
        return IfThen(
            compile_expression(argument, names, location),
            compile_expression(node["then"], names, f"{where}.then"),
            compile_expression(node["else"], names, f"{where}.else") if "else" in node else _NONE,
        )
    if head == "map":
        table = node.get("table")
        if not isinstance(table, Mapping):
            raise MappingError(f"{where}: map requires a table mapping")
        return MapTable(
            compile_expression(argument, names, location),
            tuple((str(key), value) for key, value in table.items()),
            compile_expression(node["default"], names, f"{where}.default") if "default" in node else _NONE,
        )
    name = _binding_name(node.get("as"), f"{where}.as")
    local = names | {name}
    return Each(
        compile_expression(argument, names, location),
        name,
        compile_expression(node["where"], local, f"{where}.where") if "where" in node else None,
        compile_expression(node["value"], local, f"{where}.value") if "value" in node else None,
    )


def compile_expression(node: Any, names: frozenset[str], where: str) -> Expression:
    """Compile one mapping node; ``names`` are the paths roots bound at this point."""

    if isinstance(node, str):
        return _compile_string(node, names, where)
    if node is None or isinstance(node, (bool, int, float)):
        return Literal(node)
    if isinstance(node, list):
        return ListOf(tuple(compile_expression(item, names, f"{where}[{index}]") for index, item in enumerate(node)))
    if isinstance(node, Mapping):
        return _compile_call(node, names, where)
    raise MappingError(f"{where}: unsupported node type {type(node).__name__}")


__all__ = [
    "BUILTINS",
    "EvaluationError",
    "Expression",
    "MappingError",
    "Scope",
    "canonical_json",
    "compile_expression",
    "is_present",
    "truth",
]
