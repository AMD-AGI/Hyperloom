# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the literal part of a JavaScript module: objects, arrays, strings, numbers, booleans and ``null``.

The sglang cookbook writes its deployment configs as JavaScript. A value only code can produce -- a function, an
identifier, a call, a spread, an interpolated template -- is never evaluated: it comes back as a
:class:`JsExpression` holding its source text, so a caller that needs a literal sees exactly where one is missing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


class JsLiteralError(ValueError):
    """The source is not JavaScript this reader can follow, or lacks the export asked for."""


@dataclass(frozen=True)
class JsExpression:
    """A value only running the code could produce, kept as its source text."""

    source: str


class _NotLiteral(Exception):
    """The value at hand is code; the caller rewinds and keeps it as an expression."""


_SPACE = re.compile(r"(?:\s+|//[^\n]*|/\*.*?\*/)+", re.DOTALL)
_NUMBER = re.compile(r"-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)(?![\w$])")
_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
_KEYWORDS = {"true": True, "false": False, "null": None}
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0", "\n": ""}
_VALUE_END = ",;)]}"
_HEX = 16


class _Reader:
    def __init__(self, text: str, position: int) -> None:
        self.text = text
        self.position = position

    def _error(self, message: str) -> JsLiteralError:
        line = self.text.count("\n", 0, self.position) + 1
        return JsLiteralError(f"{message} at line {line}: {self.text[self.position : self.position + 60]!r}")

    def _peek(self) -> str:
        self.skip_space()
        return self.text[self.position : self.position + 1]

    def skip_space(self) -> None:
        match = _SPACE.match(self.text, self.position)
        if match:
            self.position = match.end()

    def value(self) -> object:
        """The value at the cursor, or the expression it is when it is not wholly a literal."""
        self.skip_space()
        start = self.position
        try:
            value = self._literal()
        except _NotLiteral:
            self.position = start
            return self._expression()
        if self._peek() not in ("", *_VALUE_END):
            self.position = start
            return self._expression()
        return value

    def _literal(self) -> object:
        char = self._peek()
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        if char in "\"'":
            return self._string()
        if char == "`":
            return self._template()
        number = _NUMBER.match(self.text, self.position)
        if number:
            self.position = number.end()
            return _number(number.group())
        word = _IDENTIFIER.match(self.text, self.position)
        if word and word.group() in _KEYWORDS:
            self.position = word.end()
            return _KEYWORDS[word.group()]
        raise _NotLiteral

    def _object(self) -> dict[str, object]:
        self.position += 1
        result: dict[str, object] = {}
        while self._peek() != "}":
            key = self._key()
            if self._peek() == "(":
                result[key] = self._expression()
            elif self._peek() == ":":
                self.position += 1
                result[key] = self.value()
            else:
                raise _NotLiteral
            if not self._separator("}"):
                break
        self._expect("}")
        return result

    def _key(self) -> str:
        if self.text.startswith("...", self.position):
            raise _NotLiteral
        char = self.text[self.position : self.position + 1]
        if char in ("'", '"'):
            return self._string()
        word = _IDENTIFIER.match(self.text, self.position) or _NUMBER.match(self.text, self.position)
        if word is None:
            raise _NotLiteral
        self.position = word.end()
        return word.group()

    def _array(self) -> list[object]:
        self.position += 1
        result: list[object] = []
        while self._peek() != "]":
            if self.text.startswith("...", self.position):
                raise _NotLiteral
            result.append(self.value())
            if not self._separator("]"):
                break
        self._expect("]")
        return result

    def _separator(self, closing: str) -> bool:
        """Consume the comma after a member; ``False`` when the container closes instead."""
        char = self._peek()
        if char == ",":
            self.position += 1
            return True
        if char == closing:
            return False
        raise _NotLiteral

    def _expect(self, char: str) -> None:
        if self._peek() != char:
            raise self._error(f"expected {char!r}")
        self.position += 1

    def _string(self) -> str:
        quote = self.text[self.position]
        self.position += 1
        out: list[str] = []
        while self.position < len(self.text):
            char = self.text[self.position]
            if char == quote:
                self.position += 1
                return "".join(out)
            if char == "\\":
                out.append(self._escape())
                continue
            if char == "\n" and quote != "`":
                break
            out.append(char)
            self.position += 1
        raise self._error("unterminated string")

    def _escape(self) -> str:
        code = self.text[self.position + 1 : self.position + 2]
        if code == "u" and self.text.startswith("{", self.position + 2):
            end = self.text.index("}", self.position)
            digits, self.position = self.text[self.position + 3 : end], end + 1
            return chr(int(digits, _HEX))
        if code in ("u", "x"):
            width = 4 if code == "u" else 2
            digits = self.text[self.position + 2 : self.position + 2 + width]
            self.position += 2 + width
            return chr(int(digits, _HEX))
        self.position += 2
        return _ESCAPES.get(code, code)

    def _template(self) -> str:
        start = self.position
        self._skip_template()
        source = self.text[start : self.position]
        if "${" in source:
            raise _NotLiteral
        self.position = start
        return self._string()

    def _skip_template(self) -> None:
        self.position += 1
        while self.position < len(self.text):
            char = self.text[self.position]
            if char == "\\":
                self.position += 2
            elif char == "`":
                self.position += 1
                return
            elif self.text.startswith("${", self.position):
                self.position += 2
                self._expression(stops=";")
                self._expect("}")
            else:
                self.position += 1
        raise self._error("unterminated template")

    def _expression(self, stops: str = ",;") -> JsExpression:
        """Skip the expression at the cursor, up to a separator or the bracket closing its container."""
        start = self.position
        depth = 0
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in "\"'":
                self._string()
                continue
            if char == "`":
                self._skip_template()
                continue
            if self.text.startswith(("//", "/*"), self.position):
                self.skip_space()
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif char in stops and depth == 0:
                break
            self.position += 1
        source = self.text[start : self.position].strip()
        if not source:
            raise self._error("expected a value")
        return JsExpression(source)


def _number(text: str) -> int | float:
    if text.lstrip("-")[:2] in ("0x", "0X"):
        return int(text, _HEX)
    return float(text) if any(char in text for char in ".eE") else int(text)


def exported_value(text: str, name: str) -> object:
    """The value of ``export const <name> = ...`` in the module ``text``."""
    match = re.search(rf"^[ \t]*export\s+const\s+{re.escape(name)}\s*=", text, re.MULTILINE)
    if match is None:
        raise JsLiteralError(f"the module exports no const {name}")
    return _Reader(text, match.end()).value()


def has_expression(value: object) -> bool:
    """Whether ``value`` holds a :class:`JsExpression` anywhere inside it."""
    if isinstance(value, JsExpression):
        return True
    if isinstance(value, Mapping):
        return any(has_expression(item) for item in value.values())
    if isinstance(value, list):
        return any(has_expression(item) for item in value)
    return False


__all__ = ["JsExpression", "JsLiteralError", "exported_value", "has_expression"]
