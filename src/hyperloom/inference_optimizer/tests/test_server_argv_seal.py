# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""There is one last writer of the server argv, and nothing writes after it.

A preflight is only worth its subprocess if it inspects the argument list the
server actually receives. The composer reaches that list through a long
sequence of in-place steps, so "the argv" is whatever the last of them left
behind -- and a step added below the seal would silently move the thing being
checked out from under it.

The checks here read the dataflow rather than the spelling. A writer is found
by where its subscript key comes from -- a call to the registry's env-name
resolver, a local bound from one, or a literal that is one of the names the
registry actually returns -- so renaming the local it holds the key in, or
reaching the mapping through another subscript, changes nothing about whether
it is seen. That matters because a lint keyed on names is passed by the one
thing it exists to catch: a writer whose author picked different names.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.framework_registry import names, server_args_env_name
from hyperloom.orchestrator.actions.executors import _grid_runner, _workload_envs
from hyperloom.orchestrator.actions.executors._server_argv import (
    add_server_arg_unless_pinned,
    config_launch_env,
    config_server_argv,
    reseal_config_argv,
    seal_server_argv,
)
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

#: The name every composer ends on.
SEAL = "seal_server_argv"

#: The registry call that is the only source of the argument env's name.
RESOLVER = "server_args_env_name"

#: Every env name the registry can return, so a literal write is recognised by
#: its value rather than by a prefix someone has to remember to keep using.
ARGUMENT_ENVS = frozenset(server_args_env_name(framework) for framework in names())

#: The composers, each of which ends on the seal.
_COMPOSERS = (
    (_workload_envs, "materialize_config_with_envs"),
    (_grid_runner, "_build_variant_yaml"),
)

#: The functions allowed to write the argument env: the two composers, the two
#: helpers they delegate a step to -- the per-flag merge every injection goes
#: through and the final guards -- all of which run above the seal, plus the
#: seal itself and the re-seal a repair is written back through.
_WRITERS = frozenset(
    {
        "materialize_config_with_envs",
        "add_server_arg_unless_pinned",
        "_finalize_framework_server_args",
        "_build_variant_yaml",
        "seal_server_argv",
        "reseal_config_argv",
    }
)

#: Root of the package the census walks.
_PACKAGE = Path(_workload_envs.__file__).parents[3]


def _function(module, name: str) -> ast.FunctionDef:
    """Return the parsed ``def`` of ``name`` in ``module``'s source."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {module.__file__}")


def _seal_call(body: list[ast.stmt]) -> tuple[int, ast.Call]:
    """Return the index and node of the sole seal call among a body's statements."""
    found = [
        (index, stmt.value)
        for index, stmt in enumerate(body)
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == SEAL
    ]
    assert len(found) == 1, f"expected exactly one top-level {SEAL} call, found {len(found)}"
    return found[0]


def _argument_env_writes(source: str) -> list[tuple[str, int]]:
    """Return ``(function, lineno)`` for every write to the framework argument env.

    A subscript assignment counts when its key is the argument env's name, and
    the key is traced to its source rather than matched by spelling: the
    resolver call itself, any local this module binds from that call, or a
    literal equal to one of the registry's env names.
    """
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    resolved: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        called = node.value.func
        name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")
        if name != RESOLVER:
            continue
        resolved.update(target.id for target in node.targets if isinstance(target, ast.Name))

    def _is_argument_env(key: ast.expr) -> bool:
        if isinstance(key, ast.Call):
            called = key.func
            return (called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")) == RESOLVER
        if isinstance(key, ast.Name):
            return key.id in resolved
        return isinstance(key, ast.Constant) and key.value in ARGUMENT_ENVS

    writes: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Subscript) and _is_argument_env(target.slice)):
                continue
            owner: ast.AST = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            writes.append((getattr(owner, "name", "<module>"), node.lineno))
    return writes


@pytest.mark.parametrize(("module", "name"), _COMPOSERS)
def test_nothing_the_seal_was_given_is_touched_again_after_it(module, name):
    """The seal is the last statement naming the mapping it was handed.

    The mapping is read off the seal call's own arguments, so this follows
    whatever the composer calls it rather than a name recorded here.
    """
    body = _function(module, name).body
    index, call = _seal_call(body)
    sealed = {node.id for argument in call.args for node in ast.walk(argument) if isinstance(node, ast.Name)}
    assert sealed, "the seal call names nothing this test could follow"
    late = [
        node.lineno
        for stmt in body[index + 1 :]
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and node.id in sealed
    ]
    assert not late, f"{name} touches {sorted(sealed)} after the seal, at line(s) {late}"


def test_every_write_to_the_argument_env_is_one_the_seal_settles():
    """No module writes the argument env outside the composers and the seal.

    A third writer would be a second last writer, and which one won would
    depend on import order rather than on anything the code states.
    """
    offenders: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if "/tests/" in str(path):
            continue
        for function, lineno in _argument_env_writes(path.read_text(encoding="utf-8")):
            if function not in _WRITERS:
                offenders.append(f"{path}:{lineno} in {function}")
    assert not offenders, "the server argument env is written outside the seal's reach: " + "; ".join(offenders)


@pytest.mark.parametrize(("module", "name"), _COMPOSERS)
def test_every_write_inside_a_composer_happens_above_the_seal(module, name):
    """Composition is what runs before the seal; nothing writes the env below it."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    seal_line = _seal_call(_function(module, name).body)[1].lineno
    late = [lineno for function, lineno in _argument_env_writes(source) if function == name and lineno > seal_line]
    assert not late, f"{name} writes the argument env after the seal at line {seal_line}, at line(s) {late}"


def test_the_census_sees_a_writer_that_renames_everything_it_touches():
    """The check is dataflow, not convention, so renaming does not hide a write.

    Both plants below are real escapes from a lint that matches key spellings
    or requires the mapping itself to be a plain name.
    """
    plant = (
        "from hyperloom.inference_optimizer.framework_registry import server_args_env_name\n"
        "\n"
        "def sneak(bench, framework):\n"
        "    slot = server_args_env_name(framework)\n"
        "    mapping = bench['envs']\n"
        "    mapping[slot] = '--late-write'\n"
        "\n"
        "def sneak_nested(bench):\n"
        "    bench['envs']['EXTRA_SGLANG_ARGS'] = '--also-late'\n"
    )

    assert [function for function, _ in _argument_env_writes(plant)] == ["sneak", "sneak_nested"]


def test_what_the_preflight_reads_back_is_what_the_launch_exports(tmp_path):
    """One rendered file answers both readers, so they cannot be given two argvs.

    The preflight reads the sealed argv out of the materialised YAML; the
    launch exports that file's benchmark envs around the server. This composes
    a real config and asserts the two readings are the same string.
    """
    source = tmp_path / "base.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/path/models/Qwen-Qwen3-8B",
                    "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256, "EXTRA_SGLANG_ARGS": "  --tp   8 "},
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()

    config = materialize_config_with_envs(source, out, extra_server_args="--attention-backend aiter")

    sealed = config_server_argv(config)
    exported = config_launch_env(config, {})[sealed.env_name]
    assert sealed.text == exported
    assert sealed.argv == tuple(exported.split())
    assert "--attention-backend" in sealed.argv


def test_an_injection_never_overwrites_a_flag_the_caller_pinned():
    """The one way to add a flag leaves an operator's own spelling standing."""
    envs = {"EXTRA_VLLM_ARGS": "--block-size 128"}

    assert (
        add_server_arg_unless_pinned(envs, "vllm", "--block-size 16", pinned_by=("block-size", "block_size")) is False
    )
    assert envs["EXTRA_VLLM_ARGS"] == "--block-size 128"


def test_an_injection_merges_onto_what_is_already_composed():
    """An unpinned flag joins the string rightmost, where last-wins reads it."""
    envs = {"EXTRA_SGLANG_ARGS": "--tp 8"}

    assert add_server_arg_unless_pinned(envs, "sglang", "--trust-remote-code", pinned_by=("trust-remote-code",)) is True
    assert envs["EXTRA_SGLANG_ARGS"] == "--tp 8 --trust-remote-code"


def test_an_injection_into_an_empty_env_is_the_whole_string():
    """The first flag composed needs no separate spelling of "nothing yet"."""
    envs: dict[str, str] = {}

    assert add_server_arg_unless_pinned(envs, "sglang", "--tp 8", pinned_by=("tp",)) is True
    assert envs["EXTRA_SGLANG_ARGS"] == "--tp 8"


def test_the_seal_returns_the_argv_it_leaves_in_the_envs():
    """What the seal writes back and what it hands the caller are one string."""
    envs = {"EXTRA_SGLANG_ARGS": "  --tp 8   --attention-backend  aiter "}
    sealed = seal_server_argv(envs, "sglang")
    assert sealed.argv == ("--tp", "8", "--attention-backend", "aiter")
    assert envs["EXTRA_SGLANG_ARGS"] == sealed.text
    assert sealed.tokenized


def test_an_empty_argument_string_leaves_no_env_behind():
    """An env holding nothing is removed, so no consumer reads an empty argv as one."""
    envs = {"EXTRA_VLLM_ARGS": "   "}
    sealed = seal_server_argv(envs, "vllm")
    assert sealed.argv == ()
    assert "EXTRA_VLLM_ARGS" not in envs


def test_the_digest_is_the_argv_and_the_framework_and_nothing_else():
    """Two spellings of one argument list key the same repair; two frameworks do not."""
    tight = seal_server_argv({"EXTRA_SGLANG_ARGS": "--tp 8"}, "sglang")
    loose = seal_server_argv({"EXTRA_SGLANG_ARGS": "  --tp   8  "}, "sglang")
    other = seal_server_argv({"EXTRA_VLLM_ARGS": "--tp 8"}, "vllm")
    assert tight.digest == loose.digest
    assert tight.digest != other.digest


def test_a_repaired_argv_reaches_the_launch_through_the_rendered_config(tmp_path):
    """A repair that is not written back to the YAML is a repair the server never sees."""
    config = tmp_path / "bench.yaml"
    config.write_text(
        yaml.safe_dump(
            {"benchmark": {"framework": "sglang", "model": "/m", "envs": {"EXTRA_SGLANG_ARGS": "--tp 8 --bogus x"}}}
        ),
        encoding="utf-8",
    )
    assert config_server_argv(config).argv == ("--tp", "8", "--bogus", "x")

    resealed = reseal_config_argv(config, "--tp 8")
    assert resealed.argv == ("--tp", "8")
    # Read back off disk, because the file is what the launch opens.
    assert config_server_argv(config).argv == ("--tp", "8")
