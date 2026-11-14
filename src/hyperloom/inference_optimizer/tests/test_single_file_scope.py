# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator-declared single-file scope: what it admits, and what it refuses.

A custom workload names one module as the only thing an agent may change. The boundary
is read from the frozen benchmark directory, told to the agents, enforced on every patch
apply, and checked once at launch so an out-of-boundary campaign costs nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors._suite.custom_workload import enforce_custom_scope
from hyperloom.orchestrator.deployment.single_file import build_single_file
from hyperloom.common.optimization_scope import (
    optimization_file,
    optimization_scope,
    scope_instructions,
)
from hyperloom.orchestrator.framework.optimization_scope import check_patch_scope

_HOOK = "def hyperloom_optimize(model):\n    return model\n"


def _declare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, body: str = _HOOK, **scope: object) -> Path:
    module = tmp_path / "hyperloom_optimize.py"
    module.write_text(body, encoding="utf-8")
    declaration = {"schema_version": 1, "file": str(module), **scope}
    (tmp_path / "optimization_scope.json").write_text(json.dumps(declaration), encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    return module


def _patch(*files: str) -> str:
    return "".join(f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-x\n+y\n" for name in files)


def test_no_declaration_declares_no_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert optimization_scope() is None
    assert scope_instructions() == ""
    check_patch_scope(tmp_path, [_patch("anything.py")])


def test_the_declared_module_and_function_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _declare(tmp_path, monkeypatch)
    scope = optimization_scope()
    assert (scope.file, scope.function) == (module.resolve(), "hyperloom_optimize")
    assert optimization_file() == module.resolve()


def test_a_declaration_may_name_its_own_entry_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch, body="def tune(model):\n    return model\n", function="tune")
    assert optimization_scope().function == "tune"


@pytest.mark.parametrize(
    ("declaration", "body"),
    [
        ({"schema_version": 99}, _HOOK),
        ({"function": "not an identifier"}, _HOOK),
    ],
)
def test_an_invalid_declaration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declaration: dict, body: str
) -> None:
    _declare(tmp_path, monkeypatch, body=body, **declaration)
    with pytest.raises(ValueError, match="Invalid single-file optimization scope"):
        optimization_scope()


def test_an_unreadable_declaration_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "optimization_scope.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Unreadable single-file optimization scope"):
        optimization_scope()


def test_a_relative_path_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "optimization_scope.json").write_text(
        json.dumps({"schema_version": 1, "file": "hyperloom_optimize.py"}), encoding="utf-8"
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid single-file optimization scope"):
        optimization_scope()


def test_a_symlink_is_not_a_regular_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary names a file; a link to one outside it is a way around the boundary."""
    real = tmp_path / "real.py"
    real.write_text(_HOOK, encoding="utf-8")
    link = tmp_path / "hyperloom_optimize.py"
    link.symlink_to(real)
    (tmp_path / "optimization_scope.json").write_text(
        json.dumps({"schema_version": 1, "file": str(link)}), encoding="utf-8"
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="existing regular Python file"):
        optimization_scope()


def test_the_agents_are_told_the_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _declare(tmp_path, monkeypatch)
    instructions = scope_instructions()
    assert str(module) in instructions
    assert "hyperloom_optimize(model)" in instructions


def test_a_patch_touching_only_the_declared_module_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch)
    check_patch_scope(tmp_path, [_patch("hyperloom_optimize.py")])


@pytest.mark.parametrize(
    "targets",
    [("other.py",), ("hyperloom_optimize.py", "other.py")],
    ids=["outside", "alongside"],
)
def test_a_patch_reaching_outside_the_boundary_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, targets: tuple[str, ...]
) -> None:
    _declare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="permits changes only to"):
        check_patch_scope(tmp_path, [_patch(*targets)])


def test_a_patch_against_another_tree_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _declare(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="permits changes only to"):
        check_patch_scope(elsewhere, [_patch("hyperloom_optimize.py")])


def test_the_launch_enforces_the_boundary_before_spending_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _declare(tmp_path, monkeypatch)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(module.parent))
    enforce_custom_scope("custom")
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="permits changes only to"):
        enforce_custom_scope("custom")


def test_another_framework_enforces_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _declare(tmp_path, monkeypatch)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    enforce_custom_scope("xdit")
    assert module.exists()


def _accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, body: str = _HOOK) -> tuple[dict, Path]:
    _declare(tmp_path, monkeypatch, body=body)
    dest = tmp_path / "out"
    dest.mkdir()
    return {"current_best": {"tput": 42.0}}, dest


def test_an_accepted_module_exports_as_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, dest = _accepted(tmp_path, monkeypatch)
    result = build_single_file(state, tmp_path, optimization_scope(), dest)
    assert result["kind"] == "single_file"
    assert result["entrypoint"] == "hyperloom_optimize.hyperloom_optimize"
    assert (dest / "hyperloom_optimize.py").read_text(encoding="utf-8") == _HOOK
    assert result["accepted_performance"] == {"tput": 42.0}


def test_nothing_accepted_exports_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, dest = _accepted(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No accepted single-file workload"):
        build_single_file({}, tmp_path, optimization_scope(), dest)


def test_a_module_without_the_entry_function_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, dest = _accepted(tmp_path, monkeypatch, body="def something_else(model):\n    return model\n")
    with pytest.raises(ValueError, match=r"Module must define hyperloom_optimize\(model\)"):
        build_single_file(state, tmp_path, optimization_scope(), dest)


def test_an_unparseable_module_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, dest = _accepted(tmp_path, monkeypatch, body="def hyperloom_optimize(model:\n")
    with pytest.raises(ValueError, match="not valid Python"):
        build_single_file(state, tmp_path, optimization_scope(), dest)


def test_a_module_importing_a_sibling_ships_broken_and_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the hook is exported, so a sibling import would not resolve at the deployment site."""
    state, dest = _accepted(tmp_path, monkeypatch, body="import helper\n" + _HOOK)
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="imports 'helper' from beside itself"):
        build_single_file(state, tmp_path, optimization_scope(), dest)


def test_a_relative_import_cannot_resolve_from_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, dest = _accepted(tmp_path, monkeypatch, body="from . import helper\n" + _HOOK)
    with pytest.raises(ValueError, match="from beside itself"):
        build_single_file(state, tmp_path, optimization_scope(), dest)


def test_a_scratch_file_nothing_imports_does_not_block_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent's leftovers are not a deliverable; only what the module imports matters."""
    state, dest = _accepted(tmp_path, monkeypatch)
    (tmp_path / "scratch_notes.py").write_text("# leftover\n", encoding="utf-8")
    assert build_single_file(state, tmp_path, optimization_scope(), dest)["status"] == "exported"


def test_a_third_party_import_is_not_a_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, dest = _accepted(tmp_path, monkeypatch, body="import torch\n" + _HOOK)
    assert build_single_file(state, tmp_path, optimization_scope(), dest)["status"] == "exported"


def test_an_unreadable_patch_refuses_rather_than_checking_what_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The apply step reads the file again; a boundary checked over the rest passes on what it never saw."""
    from hyperloom.orchestrator.actions.executors._patch_sources import _resolve_framework_root

    module = _declare(tmp_path, monkeypatch)
    inside = tmp_path / "inside.patch"
    inside.write_text(_patch(module.name), encoding="utf-8")
    missing = tmp_path / "vanished.patch"

    assert _resolve_framework_root(str(tmp_path), [inside]) == tmp_path.resolve()
    assert _resolve_framework_root(str(tmp_path), [inside, missing]) is None


def test_an_unreadable_patch_without_a_declared_boundary_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No declaration means no boundary to protect; the refusal must not fire at all."""
    from hyperloom.orchestrator.actions.executors._patch_sources import (
        _resolve_framework_root,
        _resolve_framework_root_unscoped,
    )

    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    patch = tmp_path / "p.patch"
    patch.write_text(_patch("anything.py"), encoding="utf-8")
    paths = [patch, tmp_path / "gone.patch"]
    assert _resolve_framework_root(str(tmp_path), paths) == _resolve_framework_root_unscoped(str(tmp_path), paths)


def _declare_aoti(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aoti: object) -> Path:
    module = tmp_path / "hyperloom_optimize.py"
    module.write_text(_HOOK, encoding="utf-8")
    (tmp_path / "optimization_scope.json").write_text(
        json.dumps({"schema_version": 2, "file": str(module), "aoti": aoti}), encoding="utf-8"
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    return module


def test_a_schema_1_declaration_declares_no_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema 1 predates the block, so a stray one must not silently enable export."""
    from hyperloom.common.optimization_scope import aoti_config

    module = tmp_path / "hyperloom_optimize.py"
    module.write_text(_HOOK, encoding="utf-8")
    (tmp_path / "optimization_scope.json").write_text(
        json.dumps({"schema_version": 1, "file": str(module), "aoti": {"enabled": True}}), encoding="utf-8"
    )
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))
    assert aoti_config() is None


def test_declared_dynamic_dimensions_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.common.optimization_scope import aoti_config

    dim = {"input": "pixel_values", "dim": 0, "min": 1, "max": 8, "derivation": "camera count on the rig"}
    _declare_aoti(tmp_path, monkeypatch, {"enabled": True, "dynamic_dims": [dim]})
    config = aoti_config()
    assert config.enabled and config.pin_all_others is True
    assert (config.dynamic_dims[0].input, config.dynamic_dims[0].max) == ("pixel_values", 8)


@pytest.mark.parametrize(
    ("aoti", "message"),
    [
        ({"enabled": "yes"}, "aoti.enabled must be a boolean"),
        ({"enabled": True, "dynamic_dims": {}}, "aoti.dynamic_dims must be a list"),
        ({"enabled": True, "dynamic_dims": [{"input": "x"}]}, "is missing"),
        (
            {"enabled": True, "dynamic_dims": [{"input": "x", "dim": 0, "min": 8, "max": 8, "derivation": "d"}]},
            "requires min < max",
        ),
        (
            {
                "enabled": True,
                "dynamic_dims": [
                    {"input": "x", "dim": 0, "min": 1, "max": 8, "derivation": "d"},
                    {"input": "x", "dim": 0, "min": 2, "max": 9, "derivation": "d"},
                ],
            },
            "twice",
        ),
        ("not an object", "aoti must be an object"),
    ],
)
def test_a_malformed_export_block_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aoti: object, message: str
) -> None:
    from hyperloom.common.optimization_scope import aoti_config

    _declare_aoti(tmp_path, monkeypatch, aoti)
    with pytest.raises(ValueError, match=message):
        aoti_config()


def test_the_launch_refuses_a_malformed_export_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing else reports it: every other caller of aoti_config swallows the error."""
    module = _declare_aoti(tmp_path, monkeypatch, {"enabled": "yes"})
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(module.parent))
    with pytest.raises(ValueError, match="aoti.enabled must be a boolean"):
        enforce_custom_scope("custom")


def test_a_declared_export_adds_its_bounds_to_the_agent_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dim = {"input": "pixel_values", "dim": 0, "min": 1, "max": 8, "derivation": "camera count"}
    _declare_aoti(tmp_path, monkeypatch, {"enabled": True, "dynamic_dims": [dim]})
    instructions = scope_instructions()
    assert "aoti_compile_and_package" in instructions
    assert "pixel_values axis 0 in [1, 8] (camera count)" in instructions


def test_a_malformed_block_does_not_take_down_prompt_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent is simply not told about an export it could not have been given valid bounds for."""
    _declare_aoti(tmp_path, monkeypatch, {"enabled": "yes"})
    instructions = scope_instructions()
    assert "Single-file optimization contract" in instructions
    assert "aoti_compile_and_package" not in instructions


def test_an_exporting_module_declares_what_its_first_call_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "import torch\n\n\ndef hyperloom_optimize(model):\n    return torch._inductor.aoti_load_package('m.pt2')\n"
    state, dest = _accepted(tmp_path, monkeypatch, body=body)
    result = build_single_file(state, tmp_path, optimization_scope(), dest)
    assert result["runtime_requirements"]["compiles_on_first_call"] is True
    assert any("aoti_first_call_compiles" in w for w in result["warnings"])


def test_a_non_exporting_module_declares_no_runtime_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, dest = _accepted(tmp_path, monkeypatch)
    assert "runtime_requirements" not in build_single_file(state, tmp_path, optimization_scope(), dest)


def test_a_kernel_patch_is_refused_for_a_single_file_workload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """That route writes whole files into the tree, outside the one editable module."""
    from hyperloom.orchestrator.kernel.apply_kernel_patch import apply_kernel_patch

    _declare(tmp_path, monkeypatch)
    result = apply_kernel_patch(object())
    assert result["error_class"] == "single_file_scope"
    assert "hyperloom_optimize.py" in result["error"]


import pytest
from pathlib import Path
from hyperloom.common.optimization_scope import (
    OptimizationScope,
    optimization_file,
    optimization_scope,
)
from hyperloom.orchestrator.framework.optimization_scope import check_patch_scope
from .conftest import _make_workload, declare, patch


def test_scope_allows_hook_only(workload, tmp_path):
    root, hook, _, _ = workload
    assert optimization_scope() == OptimizationScope(file=hook, function="hyperloom_optimize")
    check_patch_scope(root, [patch(hook.name)])
    for name in ("benchmark.py", "extra.py", "../transformers/model.py"):
        with pytest.raises(ValueError):
            check_patch_scope(root, [patch(name)])
    with pytest.raises(ValueError):
        check_patch_scope(tmp_path, [patch(hook.name)])


def test_unsupported_schema_version_is_refused(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=3)

    with pytest.raises(ValueError, match="Invalid single-file optimization scope"):
        optimization_file()


def test_launch_scope_check_accepts_the_declared_hook(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors._suite.custom_workload import enforce_custom_scope

    root, hook, _, _ = workload
    assert optimization_scope() == OptimizationScope(file=hook, function="hyperloom_optimize")
    enforce_custom_scope("custom")
    enforce_custom_scope("sglang")


def test_kernel_direct_write_is_refused(workload, tmp_path):
    from hyperloom.orchestrator.kernel.apply_kernel_patch import KernelPatchRequest, apply_kernel_patch

    _, hook, _, _ = workload
    before = hook.read_bytes()
    result = apply_kernel_patch(KernelPatchRequest(patch_path=hook, target_file=hook, backup_root=tmp_path / "backup"))
    assert result["error_class"] == "single_file_scope"
    assert hook.read_bytes() == before


@pytest.mark.parametrize(
    "scope",
    [
        {"schema_version": 2, "function": "not an identifier"},
        {"schema_version": 3, "function": "apply"},
    ],
)
def test_scope_refuses_an_invalid_declaration(tmp_path, monkeypatch, scope):
    _make_workload(tmp_path, monkeypatch, "encoder_opt.py", "apply", scope)
    with pytest.raises(ValueError, match="Invalid"):
        optimization_scope()


def test_launch_scope_check_refuses_an_out_of_boundary_root(workload, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._suite.custom_workload import enforce_custom_scope

    _, hook, _, _ = workload
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="permits changes only"):
        enforce_custom_scope("custom")


def test_integration_cannot_use_recorded_foreign_root(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors._patch_sources import _resolve_framework_root

    root, hook, _, _ = workload
    assert _resolve_framework_root(str(root), patch_texts=[patch(hook.name)], recorded_root=str(root)) == root
    assert _resolve_framework_root(str(root), patch_texts=[patch("extra.py")], recorded_root=str(root)) is None
    assert _resolve_framework_root(str(tmp_path), patch_texts=[patch(hook.name)], recorded_root=str(tmp_path)) is None


def test_specialist_inherits_scope_and_writable_cli_config(workload, monkeypatch):
    from hyperloom.orchestrator.actions.executors._ray_runtime import safe_runtime_env
    from hyperloom.orchestrator.specialists.agent_cli_config import _build_specialist_env

    _, _, _, state = workload
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/scratch/run/claude")
    env = _build_specialist_env()
    assert env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
    worker_env = safe_runtime_env()["env_vars"]
    assert worker_env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert worker_env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"


def test_an_unreadable_scope_refuses_a_kernel_patch_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``optimization_scope`` raises on a malformed declaration, and the only ``try`` in
    ``apply_kernel_patch`` catches ``_PatchRefused``. Letting that out would crash the
    executor where every other failure on this route returns a result."""
    from hyperloom.orchestrator.kernel.apply_kernel_patch import apply_kernel_patch

    (tmp_path / "optimization_scope.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(tmp_path))

    result = apply_kernel_patch(object())

    assert result["status"] == "failed"
    assert result["error_class"] == "single_file_scope"
    assert "unreadable" in result["error"]


def test_a_confined_specialist_reads_the_declaration_and_edits_its_worktree_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module's directory is the framework repository, whose git admin the sandbox shows read-only: granting it
    writable makes the isolation refuse every specialist. The agent edits the worktree copy and delivers a diff."""
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import _section_source_hint
    from hyperloom.orchestrator.specialists.runner import _scope_readable

    module = _declare(tmp_path, monkeypatch)

    assert _scope_readable() == (tmp_path,)
    hint = "\n".join(
        _section_source_hint(
            SimpleNamespace(
                session_framework_tree="", framework_source_roots=(), source_hint_directories=(), worktree_base=""
            )
        )
    )
    assert f"copy of that file at `{module.name}`" in hint


def test_a_session_without_a_declaration_names_no_worktree_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import _section_source_hint

    monkeypatch.delenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", raising=False)
    hint = "\n".join(
        _section_source_hint(
            SimpleNamespace(
                session_framework_tree="", framework_source_roots=(), source_hint_directories=(), worktree_base=""
            )
        )
    )
    assert "copy of that file" not in hint
