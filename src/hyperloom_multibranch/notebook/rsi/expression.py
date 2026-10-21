# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frontier scorers written as expressions: what the scheduler's LLM writes, safe to run because it is not code.

An expression is one Python expression over :data:`FEATURES`, each a finite number read off the branch as it stood.
It may use numbers, ``+ - * / **``, comparisons, ``and``, ``or`` and ``not`` (true counts 1 and false 0),
``a if test else b`` and the functions of :data:`FUNCTIONS`. :meth:`Expression.read` refuses anything else, and an
expression longer than :data:`MAX_CHARS` characters or of more than :data:`MAX_NODES` syntax nodes. A division by
zero, a power out of range, a function outside its domain or a value that is not finite fails the estimate, and the
session's guard scores that branch with ``frontier-v1`` instead.

:class:`ExpressionScorer` estimates the expression's value, with basis ``custom``, its components the features it read
but those the envelope writes.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.notebook.rsi.archive import TreeBranch
from hyperloom_multibranch.notebook.rsi.frontier import (
    HOUR,
    RESERVED_COMPONENTS,
    SCORER_LABEL,
    Estimate,
    Readings,
    ScoringContext,
    is_young,
    remaining_prior,
)

MAX_CHARS = 400
MAX_NODES = 80
#: Each feature an expression may read, and what it is.
FEATURES: Mapping[str, str] = {
    "trend_pct": "the cumulative gain the branch reached over the last window, in points over the baseline",
    "gain_pct": "the branch's cumulative gain over the baseline, in points",
    "fork_gain_pct": "the gain it forked at",
    "branch_gain_pct": "what it added since it forked: gain_pct - fork_gain_pct",
    "tests": "its tests measured",
    "ended_tests": "its explore tests that got a result, as the judge's grace counts them",
    "judged_tests": "its tests judged",
    "commits": "its commits",
    "gpu_hours": "the GPU hours its tests spent",
    "age_s": "seconds since it opened",
    "since_last_commit_s": "seconds since its latest commit, or since it opened",
    "gain_per_test": "branch_gain_pct per measured test",
    "observing": "1 while a kernel-layer direction it tested lately may still pay off, else 0",
    "young": "1 while the new-branch floor holds it up, else 0",
    "prior_remaining_pct": "what similar archived branches still gained after as many tests; 0 without a prior",
    "prior_effective_sessions": "the archived sessions behind that prior, each counted by its similarity",
    "pending_pct": "the gain of its best candidate waiting for its confirm; 0 without one",
    "headroom_pct": "how far its head is below the roofline bound; 0 when the bound is unknown",
    "has_headroom": "1 when the roofline bound is known, else 0",
    "amdahl_ceiling_pct": "the most that speeding up its head's largest op could gain; 0 when the head was not profiled",
    "has_amdahl": "1 when its head was profiled, else 0",
    "open_branches": "the session's specialist branches open then",
    "best_gain_pct": "the best gain any branch of the session had reached",
    "behind_best_pct": "best_gain_pct - gain_pct",
    "elapsed_share": "the share of the session's budget its clock had spent, 0 to 1",
}


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


#: The functions an expression may call, with the number of arguments each takes, at least and at most.
FUNCTIONS: Mapping[str, tuple[Callable[..., float], int, int]] = {
    "min": (min, 2, 8),
    "max": (max, 2, 8),
    "abs": (abs, 1, 1),
    "sqrt": (math.sqrt, 1, 1),
    "log1p": (math.log1p, 1, 1),
    "clip": (_clip, 3, 3),
}
_BINARY: Mapping[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: math.pow,
}
_COMPARE: Mapping[type[ast.cmpop], Callable[[float, float], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_UNARY: Mapping[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: lambda value: float(not value),
}
_SYNTAX = (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name, ast.Constant)
_TOKENS = (ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.Load)


class ExpressionError(ValueError):
    """An expression the syntax refuses, or one that has no finite value for a branch."""


def _refused(node: ast.AST, called: set[int]) -> str:
    """Why ``node`` may not be in an expression, empty when it may; ``called`` holds the names that are called."""

    if isinstance(node, ast.BinOp) and type(node.op) not in _BINARY:
        return f"the operator {type(node.op).__name__}"
    if isinstance(node, ast.Compare) and any(type(item) not in _COMPARE for item in node.ops):
        return "that comparison"
    if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
        return f"the constant {node.value!r}"
    if isinstance(node, ast.Name) and id(node) in called:
        return "" if node.id in FUNCTIONS else f"a call of {node.id}"
    if isinstance(node, ast.Name):
        return "" if node.id in FEATURES else f"the name {node.id}, which is no feature"
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            return "a call that is not a function of FUNCTIONS by name"
        spec = FUNCTIONS.get(node.func.id)
        if spec is not None and not spec[1] <= len(node.args) <= spec[2]:
            return f"{node.func.id} of {len(node.args)} arguments"
        return "" if all(not isinstance(arg, ast.Starred) for arg in node.args) else "a starred argument"
    return "" if isinstance(node, _SYNTAX + _TOKENS) else f"{type(node).__name__}"


def _evaluate(node: ast.expr, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        return float(_UNARY[type(node.op)](_evaluate(node.operand, values)))
    if isinstance(node, ast.BinOp):
        return float(_BINARY[type(node.op)](_evaluate(node.left, values), _evaluate(node.right, values)))
    if isinstance(node, ast.BoolOp):
        truths = (_evaluate(item, values) != 0.0 for item in node.values)
        return float(all(truths) if isinstance(node.op, ast.And) else any(truths))
    if isinstance(node, ast.Compare):
        sides = [_evaluate(item, values) for item in (node.left, *node.comparators)]
        return float(all(_COMPARE[type(op)](a, b) for op, a, b in zip(node.ops, sides, sides[1:])))
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body if _evaluate(node.test, values) != 0.0 else node.orelse, values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = FUNCTIONS[node.func.id][0]
        return float(function(*(_evaluate(arg, values) for arg in node.args)))
    raise ExpressionError(f"{type(node).__name__} has no value")


@dataclass(frozen=True)
class Expression:
    """An expression :meth:`read` accepted: its ``text``, its syntax tree, and the features it reads."""

    text: str
    body: ast.expr = field(repr=False, compare=False)
    names: frozenset[str] = frozenset()

    @classmethod
    def read(cls, text: str) -> Expression:
        """``text`` as an expression; raises :class:`ExpressionError` saying what the syntax refuses."""

        source = text.strip()
        if not source or len(source) > MAX_CHARS:
            raise ExpressionError(f"an expression holds 1 to {MAX_CHARS} characters, not {len(source)}")
        try:
            body = ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(f"not an expression: {exc.msg}") from None
        nodes = list(ast.walk(body))
        if len(nodes) > MAX_NODES:
            raise ExpressionError(f"an expression holds at most {MAX_NODES} syntax nodes, not {len(nodes)}")
        called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
        refused = sorted({why for why in (_refused(node, called) for node in nodes) if why})
        if refused:
            raise ExpressionError(f"an expression may not hold {'; '.join(refused)}")
        names = frozenset(node.id for node in nodes if isinstance(node, ast.Name) and id(node) not in called)
        return cls(source, body, names)

    def value(self, values: Mapping[str, float]) -> float:
        """The expression's value with ``values`` for its features; raises :class:`ExpressionError` when it has no
        finite one."""

        try:
            found = _evaluate(self.body, values)
        except (ArithmeticError, ValueError) as exc:
            raise ExpressionError(f"{self.text}: {exc}") from None
        if not math.isfinite(found):
            raise ExpressionError(f"{self.text} is {found}, not a finite number")
        return found


def _open_branches(context: ScoringContext, t_s: float) -> int:
    tree = context.tree
    if tree is None:
        return 0
    return sum(
        1
        for item in tree.branches
        if item.name != INTEGRATION_BRANCH and item.opened_s <= t_s and (item.closed_s is None or item.closed_s > t_s)
    )


def features(branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> dict[str, float]:
    """Every feature of :data:`FEATURES`, for ``branch`` as it stood at ``t_s``."""

    policy = context.policy
    gain = branch.gain_at(t_s)
    tests = len(branch.spent_times)
    landed = [commit.t_s for commit in branch.commits if commit.t_s <= t_s]
    ended = branch.ended_by(t_s)
    judged = branch.judged_by(t_s)
    prior = remaining_prior(context.history, context.task, tests, min_similarity=policy.min_similarity)
    tree = context.tree
    best = max((item.gain_at(t_s) for item in tree.branches), default=gain) if tree is not None else gain
    budget = tree.budget_seconds if tree is not None else 0.0
    return {
        "trend_pct": gain - branch.gain_at(t_s - policy.window_seconds),
        "gain_pct": gain,
        "fork_gain_pct": branch.fork_gain_pct,
        "branch_gain_pct": gain - branch.fork_gain_pct,
        "tests": float(tests),
        "ended_tests": float(ended if ended is not None else judged if judged is not None else branch.tests_by(t_s)),
        "judged_tests": float(judged or 0),
        "commits": float(len(landed)),
        "gpu_hours": branch.gpu_seconds / HOUR,
        "age_s": t_s - branch.opened_s,
        "since_last_commit_s": t_s - max(landed, default=branch.opened_s),
        "gain_per_test": (gain - branch.fork_gain_pct) / max(tests, 1),
        "observing": float(readings.observing),
        "young": float(is_young(branch, t_s, policy)),
        "prior_remaining_pct": prior.remaining_pct if prior is not None else 0.0,
        "prior_effective_sessions": prior.effective_sessions if prior is not None else 0.0,
        "pending_pct": readings.pending_pct or 0.0,
        "headroom_pct": readings.headroom_pct or 0.0,
        "has_headroom": float(readings.headroom_pct is not None),
        "amdahl_ceiling_pct": readings.amdahl_pct or 0.0,
        "has_amdahl": float(readings.amdahl_pct is not None),
        "open_branches": float(_open_branches(context, t_s)),
        "best_gain_pct": max(best, gain),
        "behind_best_pct": max(best, gain) - gain,
        "elapsed_share": min(1.0, max(0.0, t_s / budget)) if budget > 0 else 0.0,
    }


@dataclass(frozen=True)
class ExpressionScorer:
    """A frontier scorer that estimates :attr:`expression`'s value; ``label`` as any scorer's."""

    label: str
    expression: Expression

    @classmethod
    def read(cls, label: str, text: str) -> ExpressionScorer:
        """The scorer of ``text`` under ``label``; raises :class:`ExpressionError` for either one refused."""

        if not SCORER_LABEL.fullmatch(label):
            raise ExpressionError(f"{label!r} is no scorer label: {SCORER_LABEL.pattern}")
        return cls(label, Expression.read(text))

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        values = features(branch, t_s, context, readings)
        read = {name: values[name] for name in sorted(self.expression.names - RESERVED_COMPONENTS)}
        return Estimate(self.expression.value(values), read, FrontierBasis.CUSTOM)


__all__ = [
    "FEATURES",
    "FUNCTIONS",
    "MAX_CHARS",
    "MAX_NODES",
    "Expression",
    "ExpressionError",
    "ExpressionScorer",
    "features",
]
