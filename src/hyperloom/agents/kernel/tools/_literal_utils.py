"""Bounded parsing helpers for trace metadata literals."""

from __future__ import annotations

import ast
from typing import Any


LITERAL_EVAL_MAX_CHARS = 8192
LITERAL_EVAL_ERRORS = (ValueError, SyntaxError, RecursionError, MemoryError, TypeError)


def safe_literal_eval(text: str) -> Any:
    """Parse a bounded Python literal value."""
    if len(text) > LITERAL_EVAL_MAX_CHARS:
        raise ValueError("literal too large")
    return ast.literal_eval(text)
