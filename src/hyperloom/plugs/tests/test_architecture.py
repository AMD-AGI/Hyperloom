# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guard for the plug kernel (4.2.5): what plugs may import, who may import hosted code."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3]
_PLUGS = _SRC / "hyperloom" / "plugs"
_STDLIB = frozenset(sys.stdlib_module_names)
_ALLOWED_PLUGS_IMPORTS = ("hyperloom.plugs", "hyperloom.common.entry_points")

# Hosted implementations, and dev-branch-only packages: nothing on main may reach them except their own owners.
_HOSTED = (
    "hyperloom.hosted",
    "hyperloom.sandbox",
    "hyperloom.measure.hosted",
    "hyperloom.measure.sandboxes",
    "hyperloom.measure.consistency",
    "hyperloom.measure.archive",
    "hyperloom.measure.runner",
    "hyperloom_multibranch",
)
# Packages that exist only on the multibranch development branch; main must stay self-contained.
_DEV_ONLY = ("hyperloom.measure", "hyperloom.sandbox", "hyperloom.hosted", "hyperloom_multibranch")
# Modules of hyperloom.hosted that main itself carries; the rest of that package is dev-only.
_HOSTED_ON_MAIN = (
    "hyperloom.hosted.campaign",
    "hyperloom.hosted.context",
    "hyperloom.hosted.entry",
    "hyperloom.hosted.host_info",
)
_E2B_OWNER = "hyperloom/sandbox/e2b.py"


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return found


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _plugs_import_violations(source: str) -> list[str]:
    bad = []
    for line, module in _imports(ast.parse(source)):
        top = module.split(".")[0]
        if top in _STDLIB or _under(module, _ALLOWED_PLUGS_IMPORTS):
            continue
        # ``from hyperloom.common import entry_points`` is reported as hyperloom.common + the joined name.
        if module == "hyperloom.common" or module == "hyperloom":
            continue
        bad.append(f"line {line}: imports {module}")
    return bad


def _hosted_import_violations(rel: str, source: str) -> list[str]:
    bad = []
    for line, module in _imports(ast.parse(source)):
        if _under(module, _HOSTED) and not _under(
            rel.removesuffix(".py").replace("/", "."), _HOSTED + ("hyperloom.plugs.builtin",)
        ):
            bad.append(f"{rel}:{line}: imports hosted implementation {module}")
        if module.split(".")[0] == "e2b" and rel != _E2B_OWNER:
            bad.append(f"{rel}:{line}: imports e2b outside {_E2B_OWNER}")
    return bad


def _plugs_modules() -> list[Path]:
    return [p for p in _PLUGS.rglob("*.py") if "tests" not in p.relative_to(_PLUGS).parts]


def _main_modules() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if "tests" not in p.relative_to(_SRC).parts and ".venv" not in p.parts]


def test_rule_1_plugs_import_only_the_standard_library_and_entry_points() -> None:
    problems = {str(p.relative_to(_SRC)): v for p in _plugs_modules() if (v := _plugs_import_violations(p.read_text()))}
    assert problems == {}


@pytest.mark.parametrize(
    "source",
    [
        "import yaml",
        "from hyperloom.common.io import read_json",
        "import hyperloom.orchestrator.loop",
        "def f():\n    from hyperloom.measure import hosted",
        "import hyperloom_multibranch",
    ],
)
def test_rule_1_guard_refuses_a_foreign_import_even_inside_a_function(source) -> None:
    (message,) = [m for m in _plugs_import_violations(source) if "imports" in m][:1]
    assert "imports" in message


def test_rule_1_guard_accepts_the_two_sanctioned_imports() -> None:
    source = "import json\nfrom hyperloom.common.entry_points import iter_entry_point_objects\nfrom hyperloom.plugs.socket import Socket"
    assert _plugs_import_violations(source) == []


def test_rule_2_no_module_imports_a_hosted_implementation_outside_its_owners() -> None:
    problems = [m for p in _main_modules() for m in _hosted_import_violations(str(p.relative_to(_SRC)), p.read_text())]
    assert problems == []


def test_rule_2_main_imports_no_package_that_exists_only_on_the_dev_branch() -> None:
    offenders = []
    for path in _main_modules():
        for line, module in _imports(ast.parse(path.read_text())):
            if _under(module, _DEV_ONLY) and not _under(module, _HOSTED_ON_MAIN):
                offenders.append(f"{path.relative_to(_SRC)}:{line}: {module}")
    assert offenders == []


def test_rule_2_guard_names_the_hosted_module_a_plug_module_imports() -> None:
    bad = _hosted_import_violations("hyperloom/plugs/host.py", "def f():\n    import hyperloom.measure.hosted\n")
    assert bad == ["hyperloom/plugs/host.py:2: imports hosted implementation hyperloom.measure.hosted"]


def test_rule_2_guard_lets_the_owners_and_builtin_import_hosted_code() -> None:
    assert _hosted_import_violations("hyperloom/hosted/entry.py", "import hyperloom.sandbox.e2b") == []
    assert _hosted_import_violations("hyperloom/plugs/builtin.py", "import hyperloom.hosted.events") == []


def test_rule_2_only_the_e2b_module_may_import_e2b() -> None:
    assert _hosted_import_violations("hyperloom/sandbox/e2b.py", "import e2b") == []
    assert _hosted_import_violations("hyperloom/sandbox/local.py", "import e2b") == [
        "hyperloom/sandbox/local.py:1: imports e2b outside hyperloom/sandbox/e2b.py"
    ]


def _calls_implementation(source: str) -> bool:
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "Implementation"
        for n in ast.walk(ast.parse(source))
    )


def test_rule_3_only_builtin_constructs_implementations() -> None:
    constructors = sorted(p.name for p in _plugs_modules() if _calls_implementation(p.read_text()))
    assert constructors == ["builtin.py"]


def test_rule_3_guard_sees_an_implementation_built_outside_builtin() -> None:
    assert _calls_implementation("x = Implementation('events', 'local', f)")
    assert _calls_implementation("x = socket.Implementation('events', 'local', f)")
    assert not _calls_implementation("x = Socket('events', object)")


def test_rule_3_builtin_registers_by_import_path_string_and_imports_no_implementation_at_module_level() -> None:
    tree = ast.parse((_PLUGS / "builtin.py").read_text())
    top_level = [m for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for _, m in _imports(node)]
    assert not [m for m in top_level if m.startswith("hyperloom.plugs.") and m.endswith("_local")]
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "hyperloom.plugs.events_local:create" in strings


def test_rule_3_importing_builtin_and_building_the_registry_does_not_load_the_implementation() -> None:
    code = (
        "import sys\n"
        "from hyperloom.plugs.builtin import builtin_registry\n"
        "r = builtin_registry()\n"
        "assert 'hyperloom.plugs.events_local' not in sys.modules\n"
        "r.implementation('events', 'local')\n"
        "assert 'hyperloom.plugs.events_local' not in sys.modules\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(_SRC)}, check=False
    )
    assert (out.returncode, out.stdout.strip()) == (0, "ok"), out.stderr
