# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of scripts/code_metrics_design.py: which design refusal fires, and when none does.

Each case runs the real diff machinery in a scratch git repository: the base is
committed, the change is left in the working tree, and the result names the check and
the line, so a refusal from another check cannot stand in for the one under test.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

import code_metrics_design as design
from code_metrics_collect import ToolError, Units

CONTRACTS = """\
[importlinter]
root_packages =
    pkg

[importlinter:contract:layers]
name = layers
type = layers
layers =
    pkg.top
    pkg.low  # a trailing comment
    pkg.common

[importlinter:contract:independent]
name = independent
type = independence
modules =
    pkg.top.alpha
    pkg.top.beta
"""

ENUMS = """\
import enum


class Status(str, enum.Enum):
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    _HIDDEN = "hidden"


class Outcome(enum.Enum):
    SKIPPED = "skipped"


class Plain:
    NAME = "plain"
"""

LOW = "src/pkg/low/mod.py"
USES_STATUS = "from pkg.common.states import Status\n"
TOP = "src/pkg/top/mod.py"


def git(root: Path, *args: str) -> None:
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "-C", str(root), *ident, *args], check=True, capture_output=True)


def write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(textwrap.dedent(text), encoding="utf-8")


def refusals(root: Path, base: dict[str, str], head: dict[str, str], max_tuple: int = 2) -> list[tuple[str, str, int]]:
    """(check, path, line) of every refusal for ``base`` committed and ``head`` written over it."""
    write(root, {"src/pkg/common/states.py": ENUMS, **base})
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    write(root, head)
    scope = design.Scope(
        roots=("src", "scripts"), exclude=(), contracts=CONTRACTS, vocabulary_roots=("src",), max_tuple=max_tuple
    )
    problems = design.design_problems(root, scope, Units(root), "HEAD")
    return sorted((p.check, p.where, p.line) for p in problems)


def messages(root: Path, base: dict[str, str], head: dict[str, str]) -> list[str]:
    write(root, {"src/pkg/common/states.py": ENUMS, **base})
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    write(root, head)
    scope = design.Scope(("src",), (), CONTRACTS, ("src",), 2)
    return [p.message for p in design.design_problems(root, scope, Units(root), "HEAD")]


# --- package units ------------------------------------------------------------------


def test_units_come_from_the_contracts_and_the_innermost_wins() -> None:
    packages = design.Packages.from_contracts(CONTRACTS)
    assert packages.unit("pkg.top.alpha.x._y") == "pkg.top.alpha"
    assert packages.unit("pkg.top.gamma") == "pkg.top"
    assert packages.unit("pkg.low") == "pkg.low"
    assert packages.unit("pkg") == "pkg"
    assert packages.unit("numpy._core") is None


def test_unreadable_contracts_are_an_error_not_an_empty_unit_list() -> None:
    with pytest.raises(ToolError, match="import-linter"):
        design.Packages.from_contracts("not an ini file")


# --- A1 private-name reach ---------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "from pkg.low.mod import _helper\n",
        "from pkg.low._internal import helper\n",
        "import pkg.low._internal.thing\n",
        "from pkg.low import mod\nx = mod._helper\n",
        "import pkg.low.mod as m\nx = m._helper()\n",
        "import pkg.low.mod\nx = pkg.low.mod._helper\n",
        "from ..low.mod import _helper\n",
    ],
)
def test_reaching_into_another_units_private_name_fails(tmp_path: Path, source: str) -> None:
    got = refusals(tmp_path, {TOP: "x = 1\n"}, {TOP: "x = 1\n" + source})
    line = len(source.splitlines()) + 1
    assert got == [("private-reach", TOP, line)]


def test_an_independent_sibling_is_another_unit(tmp_path: Path) -> None:
    alpha = "src/pkg/top/alpha/a.py"
    assert refusals(tmp_path, {alpha: ""}, {alpha: "from pkg.top.beta.b import _x\n"}) == [("private-reach", alpha, 1)]


@pytest.mark.parametrize(
    "source",
    [
        "from pkg.top.other import _helper\n",  # same unit
        "from pkg.low.mod import __version__\n",  # dunder
        "from pkg.low.mod import helper\n",  # public
        "from numpy._core import thing\n",  # not one of ours
        "from pkg.low.mod import helper\nx = self._cache\n",  # not through an import
    ],
)
def test_same_unit_dunders_public_and_foreign_names_pass(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: source}) == []


def test_a_local_name_shadowing_an_import_is_not_the_import(tmp_path: Path) -> None:
    head = (
        "from pkg.low.mod import obj\n\n\n"
        "def f(obj):\n    return obj._cache\n\n\n"
        "def g():\n    obj = make()\n    return lambda: obj._cache\n\n\n"
        "def h():\n    return obj._cache\n"
    )
    assert refusals(tmp_path, {TOP: ""}, {TOP: head}) == [("private-reach", TOP, 14)]


def test_tests_may_reach_private_names(tmp_path: Path) -> None:
    path = "src/pkg/top/tests/test_mod.py"
    assert refusals(tmp_path, {path: ""}, {path: "from pkg.low.mod import _helper\n"}) == []


def test_a_moved_private_import_is_not_added(tmp_path: Path) -> None:
    other = "src/pkg/top/other.py"
    base = {TOP: "from pkg.low.mod import _helper\n", other: "y = 2\n"}
    assert refusals(tmp_path, base, {TOP: "x = 1\n", other: "from pkg.low.mod import _helper\ny = 2\n"}) == []


def test_an_existing_private_import_is_backlog(tmp_path: Path) -> None:
    base = {TOP: "from pkg.low.mod import _helper\n"}
    assert refusals(tmp_path, base, {TOP: "from pkg.low.mod import _helper\nx = 1\n"}) == []


def test_the_private_reach_message_names_the_rule_and_the_fix(tmp_path: Path) -> None:
    (message,) = messages(tmp_path, {TOP: ""}, {TOP: "from pkg.low.mod import _helper\n"})
    assert message.startswith("A1 private-name reach: `pkg.low.mod._helper` uses `_helper`, private to `pkg.low`")
    assert "make it public in its owner" in message


# --- A8 flag argument ---------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "def f(x, fast: bool):\n    if fast:\n        return x\n    return x * 2\n",
        "def f(x, *, fast=False):\n    '''Doc.'''\n    if not fast:\n        raise ValueError(x)\n    return x\n",
        "def f(x, mode: Literal['a', 'b']):\n    if mode == 'a':\n        return 1\n    return 2\n",
        "def f(x, mode: 'bool'):\n    if mode:\n        return 1\n    return 2\n",
        "def f(x, mode: typing.Literal['a', 'b']):\n    if 'a' == mode:\n        return 1\n    return 2\n",
    ],
)
def test_a_new_function_whose_flag_picks_its_body_fails(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {LOW: "x = 1\n"}, {LOW: "x = 1\n" + source}) == [("flag-argument", LOW, 2)]


@pytest.mark.parametrize(
    "source",
    [
        # The branch does not end the function: one body with an optional step.
        "def f(x, fast: bool):\n    if fast:\n        x += 1\n    return x\n",
        # The flag is not the first statement.
        "def f(x, fast: bool):\n    y = x\n    if fast:\n        return y\n    return x\n",
        # Three values are a choice, not a flag.
        "def f(x, mode: Literal['a', 'b', 'c']):\n    if mode == 'a':\n        return 1\n    return 2\n",
        # Not a flag parameter.
        "def f(x, items: list):\n    if items:\n        return 1\n    return 2\n",
        # CLI entry points.
        "def main(verbose: bool = False):\n    if verbose:\n        return 1\n    return 0\n",
        "@app.command()\ndef run(dry: bool = False):\n    if dry:\n        return 1\n    return 0\n",
        "def handle(args: argparse.Namespace, dry: bool):\n    if dry:\n        return 1\n    return 0\n",
    ],
)
def test_optional_steps_late_branches_choices_and_cli_entry_points_pass(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == []


def test_an_existing_function_that_gains_a_flag_is_not_new(tmp_path: Path) -> None:
    base = {LOW: "def f(x):\n    return x\n"}
    head = {LOW: "def f(x, fast: bool = False):\n    if fast:\n        return x\n    return x\n"}
    assert refusals(tmp_path, base, head) == []


@pytest.mark.parametrize("path", ["src/pkg/low/tests/test_x.py", "src/pkg/low/test_x.py", "src/pkg/low/conftest.py"])
def test_a_flag_argument_in_a_test_passes(tmp_path: Path, path: str) -> None:
    assert refusals(tmp_path, {}, {path: "def f(fast: bool):\n    if fast:\n        return 1\n    return 2\n"}) == []


# --- A9 tuple returns -------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "def f() -> tuple[int, str, bytes]:\n    return g()\n",
        "def f() -> 'Tuple[int, int, int]':\n    return g()\n",
        "def f() -> tuple[int, int, int] | None:\n    return None\n",
        "def f() -> Optional[tuple[int, int, int]]:\n    return None\n",
        "def f() -> typing.Union[str, Tuple[int, int, int]]:\n    return ''\n",
        "def f(x):\n    if x:\n        return 1, 2, 3\n    return None\n",
    ],
)
def test_a_new_function_returning_three_or_more_positions_fails(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == [("tuple-return", LOW, 1)]


@pytest.mark.parametrize(
    "source",
    [
        "def f() -> tuple[int, str]:\n    return 1, 'a'\n",
        "def f() -> tuple[int, ...]:\n    return (1, 2, 3, 4)[:1]\n",
        "def f():\n    def g():\n        return 1, 2, 3\n    return g\n",  # g is new too, but is the one returning
        "def f():\n    return [1, 2, 3]\n",
        "def f() -> Optional[tuple[int, int]]:\n    return None\n",
    ],
)
def test_pairs_variadic_tuples_and_other_shapes_pass(tmp_path: Path, source: str) -> None:
    got = refusals(tmp_path, {LOW: ""}, {LOW: source})
    assert all(line != 1 for _check, _path, line in got)


def test_a_protocol_method_returns_the_tuple_its_protocol_fixes(tmp_path: Path) -> None:
    source = "class C:\n    def __reduce__(self):\n        return (C, (), {})\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == []


def test_a_display_returned_as_a_variadic_tuple_passes(tmp_path: Path) -> None:
    source = "def f() -> tuple[int, ...]:\n    return (1, 2, 3)\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == []
    fixed = "def f() -> tuple[int, int]:\n    return (1, 2, 3)\n"
    assert refusals(tmp_path / "fixed", {LOW: ""}, {LOW: fixed}) == [("tuple-return", LOW, 1)]


def test_a_variadic_tuple_has_no_fixed_number_of_positions(tmp_path: Path) -> None:
    source = "def f() -> tuple[int, ...]:\n    return g()\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}, max_tuple=1) == []


def test_the_tuple_limit_comes_from_the_config(tmp_path: Path) -> None:
    source = "def f() -> tuple[int, str, bytes]:\n    return g()\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}, max_tuple=3) == []


def test_an_existing_function_returning_a_triple_is_backlog(tmp_path: Path) -> None:
    base = {LOW: "def f():\n    return 1, 2, 3\n"}
    assert refusals(tmp_path, base, {LOW: "def f():\n    x = 0\n    return x, 2, 3\n"}) == []


# --- A10 raw vocabulary -----------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "ok = status == 'running'\n",
        "ok = 'done' != status\n",
        "ok = status in ('queued', 'done')\n",
        "ok = status not in {'running'}\n",
    ],
)
def test_comparing_with_an_enum_value_fails(tmp_path: Path, source: str) -> None:
    head = {TOP: USES_STATUS + source}
    assert refusals(tmp_path, {TOP: ""}, head) == [("raw-vocabulary", TOP, 2)]


def test_the_enum_may_be_named_through_its_module_or_defined_here(tmp_path: Path) -> None:
    head = {TOP: "from pkg.common import states\nx = states.Status\nok = status == 'running'\n"}
    assert refusals(tmp_path, {TOP: ""}, head) == [("raw-vocabulary", TOP, 3)]
    local = "class Mode(Enum):\n    FAST = 'fast'\n\n\nok = mode == 'fast'\n"
    assert refusals(tmp_path / "local", {}, {TOP: local}) == [("raw-vocabulary", TOP, 5)]


def test_a_case_pattern_is_a_comparison(tmp_path: Path) -> None:
    source = "match status:\n    case 'unknown':\n        pass\n    case 'done' | 'running':\n        pass\n"
    got = refusals(tmp_path, {TOP: ""}, {TOP: USES_STATUS + source})
    assert got == [("raw-vocabulary", TOP, 5), ("raw-vocabulary", TOP, 5)]


@pytest.mark.parametrize(
    "source",
    [
        "ok = run.status.value == 'running'\n",  # a member's value
        "ok = getattr(run.status, 'value', None) == 'running'\n",
        "ok = job.run_status == 'running'\n",  # named for the enum's last words
    ],
)
def test_a_members_value_or_a_name_for_the_vocabulary_is_the_enum(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: USES_STATUS + source}) == [("raw-vocabulary", TOP, 2)]


@pytest.mark.parametrize(
    "source",
    [
        "def parse(raw):\n    if raw == 'running':\n        return Status.RUNNING\n",  # the conversion
        "ok = v != 'running'\n",
        "ok = str(x or '') in {'running', 'kept'}\n",
        "ok = status_text == 'running'\n",  # not the enum's last words
    ],
)
def test_raw_input_on_its_way_to_the_enum_passes(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: USES_STATUS + source}) == []


def test_a_module_that_never_names_the_enum_compares_another_vocabulary(tmp_path: Path) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: "ok = row.get('status') == 'running'\n"}) == []


def test_one_enum_defined_in_two_modules_is_still_one_vocabulary(tmp_path: Path) -> None:
    copy = {"src/pkg/low/states.py": "import enum\n\n\nclass Status(enum.Enum):\n    RUNNING = 'running'\n"}
    head = {TOP: USES_STATUS + "ok = status == 'running'\n"}
    assert refusals(tmp_path, {TOP: "", **copy}, head) == [("raw-vocabulary", TOP, 2)]


def test_a_value_two_enums_share_is_not_one_vocabulary(tmp_path: Path) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: USES_STATUS + "ok = status == 'skipped'\n"}) == []


@pytest.mark.parametrize(
    "source",
    [
        "ok = status == 'queued'\n",  # not an enum value
        "ok = status == 'plain'\n",  # a class that is not an enum
        "ok = status == 'hidden'\n",  # a private member
        "ok = 'running' in text\n",  # a substring or key test, not a vocabulary comparison
        "ok = status is Status.RUNNING\n",
        "x = {'running': 1}\n",
    ],
)
def test_other_strings_and_enum_members_pass(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {TOP: ""}, {TOP: USES_STATUS + source}) == []


def test_raw_vocabulary_in_a_test_passes(tmp_path: Path) -> None:
    path = "src/pkg/top/tests/test_states.py"
    assert refusals(tmp_path, {}, {path: USES_STATUS + "assert status == 'running'\n"}) == []


# --- A11 hidden global write ------------------------------------------------------


def test_an_added_global_statement_fails(tmp_path: Path) -> None:
    head = {LOW: "CACHE = None\n\n\ndef load():\n    global CACHE\n    CACHE = 1\n"}
    assert refusals(tmp_path, {LOW: "CACHE = None\n"}, head) == [("global-write", LOW, 5)]


def test_a_store_into_globals_fails(tmp_path: Path) -> None:
    head = {LOW: "def load():\n    globals()['CACHE'] = 1\n    return globals()['CACHE']\n"}
    assert refusals(tmp_path, {LOW: ""}, head) == [("global-write", LOW, 2)]


def test_a_nonlocal_closure_passes(tmp_path: Path) -> None:
    source = "def f():\n    n = 0\n\n    def g():\n        nonlocal n\n        n += 1\n\n    g()\n    return n\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == []


def test_an_existing_global_and_one_in_a_test_pass(tmp_path: Path) -> None:
    source = "def load():\n    global CACHE\n    CACHE = 1\n"
    test = "src/pkg/low/tests/test_cache.py"
    assert refusals(tmp_path, {LOW: source}, {LOW: source + "X = 1\n", test: source}) == []


# --- A12 forwarders -----------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "def f(a, b):\n    return g(a, b)\n",
        "def f(a, *args, **kwargs):\n    '''Doc.'''\n    return g(a, *args, **kwargs)\n",
        "def f(a, *, b):\n    return mod.g(a, b=b)\n",
        "async def f(a):\n    return await g(a)\n",
        "class C:\n    def f(self, a):\n        return self._impl.f(a)\n",
        "class C:\n    def f(self, a):\n        return self._impl.next(a)\n",  # a method named like a builtin
        "class C:\n    @classmethod\n    def f(cls, a):\n        return cls.g(a)\n",  # no decorator of its own
    ],
)
def test_a_new_function_that_only_passes_its_parameters_on_fails(tmp_path: Path, source: str) -> None:
    got = refusals(tmp_path, {LOW: ""}, {LOW: source})
    assert [(check, path) for check, path, _line in got] == [("forwarder", LOW)]


@pytest.mark.parametrize(
    "source",
    [
        "def f(a, b):\n    return g(a, b, 1)\n",  # adds an argument
        "def f(a, b):\n    return g(a)\n",  # drops one
        "def f(a, b):\n    return g(b=a, a=b)\n",  # renames
        "def f(a):\n    return g(a + 1)\n",  # converts
        "def f():\n    return g()\n",  # no parameters to pass through
        "def f(fn, a):\n    return fn(a)\n",  # calls a parameter
        "def f(path):\n    return str(path)\n",  # a builtin conversion
        "def f(state):\n    return Executor(state)\n",  # a factory
        "def f(kind):\n    return HANDLERS.get(kind)\n",  # a table lookup
        "def f(*parts):\n    return ', '.join(parts)\n",  # a method of a constant
        "def deco(fn):\n    @functools.wraps(fn)\n    def inner(*a, **k):\n        return fn(*a, **k)\n\n    return inner\n",
        "@functools.lru_cache(maxsize=128)\ndef parse(key):\n    return parse_uncached(key)\n",  # adds a cache
        "@registry.register('x')\ndef run(job):\n    return execute(job)\n",  # registers
        "def f(a):\n    x = g(a)\n    return x\n",
        "class C:\n    def __len__(self):\n        return len(self)\n",  # a protocol method
        "class C:\n    @abstractmethod\n    def f(self, a):\n        return g(a)\n",
    ],
)
def test_functions_that_do_something_with_their_parameters_pass(tmp_path: Path, source: str) -> None:
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == []


def test_a_method_overriding_its_base_may_delegate(tmp_path: Path) -> None:
    base = {"src/pkg/low/base.py": "class Base(ABC):\n    @abstractmethod\n    def submit(self, job): ...\n"}
    source = "from pkg.low.base import Base\n\n\nclass Impl(Base):\n    def submit(self, job):\n        return self._client.submit(job)\n"
    assert refusals(tmp_path, base, {LOW: source}) == []
    extra = source + "\n    def other(self, job):\n        return self._client.other(job)\n"
    assert refusals(tmp_path / "extra", base, {LOW: extra}) == [("forwarder", LOW, 8)]


def test_a_static_method_has_no_receiver_to_drop(tmp_path: Path) -> None:
    source = "class C:\n    @staticmethod\n    def f(self, a):\n        return g(self, a)\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: source}) == [("forwarder", LOW, 3)]


def test_a_forwarder_in_a_test_passes(tmp_path: Path) -> None:
    path = "src/pkg/low/tests/test_f.py"
    assert refusals(tmp_path, {}, {path: "def f(a):\n    return g(a)\n"}) == []


# --- A12 single implementation ----------------------------------------------------


def test_a_new_abc_with_one_implementation_in_its_unit_fails(tmp_path: Path) -> None:
    head = {LOW: "class Port(ABC):\n    pass\n\n\nclass Impl(Port):\n    pass\n"}
    assert refusals(tmp_path, {LOW: ""}, head) == [("single-implementation", LOW, 1)]


def test_a_new_protocol_counts_structural_implementations(tmp_path: Path) -> None:
    proto = "class Sink(Protocol):\n    def write(self, x): ...\n"
    one = "class File:\n    def write(self, x):\n        return x\n"
    other = "src/pkg/low/other.py"
    assert refusals(tmp_path, {LOW: ""}, {LOW: proto + "\n\n" + one}) == [("single-implementation", LOW, 1)]
    head = {LOW: proto + "\n\n" + one, other: one.replace("File", "Socket")}
    assert refusals(tmp_path / "two", {LOW: ""}, head) == []


def test_an_interface_a_lower_unit_declares_for_a_higher_one_passes(tmp_path: Path) -> None:
    common = "src/pkg/common/ports.py"
    head = {
        common: "class Port(ABC):\n    pass\n",
        TOP: "from pkg.common.ports import Port\n\n\nclass Impl(Port):\n    pass\n",
    }
    assert refusals(tmp_path, {}, head) == []


def test_an_abc_with_two_implementations_or_none_or_an_existing_one_passes(tmp_path: Path) -> None:
    two = "class Port(metaclass=ABCMeta):\n    pass\n\n\nclass A(Port):\n    pass\n\n\nclass B(Port):\n    pass\n"
    assert refusals(tmp_path, {LOW: ""}, {LOW: two}) == []
    existing = "class Port(ABC):\n    pass\n"
    assert refusals(tmp_path / "e", {LOW: existing}, {LOW: existing + "\n\nclass Impl(Port):\n    pass\n"}) == []


# --- the diff ---------------------------------------------------------------------


def test_a_renamed_module_is_not_new_code(tmp_path: Path) -> None:
    source = "from pkg.low.mod import _h\n\n\ndef f(a):\n    return g(a)\n\n\ndef t() -> tuple[int, int, int]:\n    return g()\n"
    old, new = "src/pkg/top/old.py", "src/pkg/top/new.py"
    write(tmp_path, {"src/pkg/common/states.py": ENUMS, old: source})
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "mv", old, new)
    scope = design.Scope(("src",), (), CONTRACTS, ("src",), 2)
    assert design.design_problems(tmp_path, scope, Units(tmp_path), "HEAD") == []


def test_an_untracked_new_module_is_judged(tmp_path: Path) -> None:
    new = "src/pkg/top/fresh.py"
    assert refusals(tmp_path, {}, {new: "def f():\n    global X\n"}) == [("global-write", new, 2)]
