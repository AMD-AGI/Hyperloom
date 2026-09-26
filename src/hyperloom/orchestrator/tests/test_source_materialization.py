# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Actual Python imports from complete accepted source bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.agents.kernel.tools.backends import geak_runner
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.source_materialization import SourceMaterializationError, materialize_source_stack
from hyperloom.orchestrator.source_snapshot import snapshot_source_layer


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _commit(root: Path) -> str:
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _repository(root: Path, package: str = "accepted_pkg", prefix: str = "python") -> Path:
    directory = root / prefix / package
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("")
    for name in ("alpha", "beta", "removed"):
        (directory / f"{name}.py").write_text(f"VALUE = 'stock-{name}'\n")
    _git(root, "init", "-q")
    _commit(root)
    return root


def _keep(root: Path, snapshot: Path, writes: dict[str, str | None], *, prefix: str = "python") -> dict:
    for relative, content in writes.items():
        target = root / relative
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    commit = _commit(root)
    snapshot_source_layer(
        framework_root=root,
        base_sha=commit,
        rel_paths=list(writes),
        dest_dir=snapshot,
        import_root=prefix,
        provenance="integrate_patch",
        declared_ops={path: "delete" if content is None else "upsert" for path, content in writes.items()},
        extra={"artifacts_outside_root": 0, "commit_semantics": "accepted"},
    )
    return {
        "scope": "source_patch",
        "variant_name": snapshot.name,
        "framework_root": str(root),
        "base_sha": commit,
        "source_import_root": prefix,
        "source_snapshot": str(snapshot),
        "source_snapshot_complete": True,
        "target_files": list(writes),
    }


def _stack(tmp_path: Path) -> tuple[Path, dict]:
    root = _repository(tmp_path / "repo")
    first = _keep(root, tmp_path / "first", {"python/accepted_pkg/alpha.py": "VALUE = 'first'\n"})
    second = _keep(
        root,
        tmp_path / "second",
        {"python/accepted_pkg/beta.py": "VALUE = 'second'\n", "python/accepted_pkg/removed.py": None},
    )
    return root, {"optimization_stack": [first, second]}


def _imports(descriptor: dict, stock: Path, cwd: Path) -> dict:
    prefixes = [str(Path(descriptor["bundle_root"]) / prefix) for prefix in descriptor["pythonpath_prefixes"]]
    script = """import importlib, json
out = {}
for name in ('alpha', 'beta', 'removed'):
    try:
        module = importlib.import_module('accepted_pkg.' + name)
        out[name] = {'value': module.VALUE, 'file': module.__file__}
    except ModuleNotFoundError:
        out[name] = None
print(json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=cwd,
        env={"PATH": os.defpath, "PYTHONPATH": os.pathsep.join([*prefixes, str(stock)])},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_two_accepted_layers_delete_and_mutable_head_are_reconstructed(tmp_path: Path) -> None:
    root, best = _stack(tmp_path)
    (root / ".gitattributes").write_text("python/accepted_pkg/alpha.py export-ignore\n")
    _commit(root)
    (root / "python/accepted_pkg/alpha.py").write_text("VALUE = 'later-unaccepted-head'\n")
    _commit(root)
    stock = _repository(tmp_path / "stock") / "python"

    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    result = _imports(descriptor, stock, tmp_path)

    assert result["alpha"]["value"] == "first"
    assert result["beta"]["value"] == "second"
    assert result["removed"] is None
    assert result["alpha"]["file"].startswith(descriptor["bundle_root"])
    manifest = json.loads((Path(descriptor["bundle_root"]) / "manifest.json").read_bytes())
    assert manifest["trees"][0]["accepted_commit"] == best["optimization_stack"][-1]["base_sha"]
    assert descriptor["required_layer_ids"] == ["first", "second"]
    assert manifest["deleted_modules"] == ["accepted_pkg.removed"]
    assert materialize_source_stack(best, tmp_path / "materialized") == descriptor


def test_bundle_relocates_without_original_repo_or_snapshots(tmp_path: Path) -> None:
    root, best = _stack(tmp_path)
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    moved = tmp_path / "moved"
    shutil.copytree(descriptor["bundle_root"], moved)
    shutil.rmtree(root)
    shutil.rmtree(tmp_path / "first")
    shutil.rmtree(tmp_path / "second")
    shutil.rmtree(tmp_path / "materialized")
    relocated = {**descriptor, "bundle_root": str(moved)}
    assert hashlib.sha256((moved / "manifest.json").read_bytes()).hexdigest() == descriptor["manifest_sha256"]
    result = _imports(relocated, _repository(tmp_path / "stock") / "python", tmp_path)
    assert result["alpha"]["value"] == "first" and result["beta"]["value"] == "second"
    assert result["removed"] is None


def test_later_overwrite_and_readdition_replace_prior_operations(tmp_path: Path) -> None:
    root, best = _stack(tmp_path)
    best["optimization_stack"].append(
        _keep(
            root,
            tmp_path / "third",
            {
                "python/accepted_pkg/alpha.py": "VALUE = 'third'\n",
                "python/accepted_pkg/removed.py": "VALUE = 'returned'\n",
            },
        )
    )
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    result = _imports(descriptor, tmp_path / "absent", tmp_path)
    assert result["alpha"]["value"] == "third"
    assert result["beta"]["value"] == "second"
    assert result["removed"]["value"] == "returned"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_payload",
        "wrong_payload",
        "missing_manifest",
        "false_complete",
        "malformed_complete",
        "missing_complete",
        "malformed_schema",
        "missing_coverage",
        "outside_artifact",
        "malformed_stack_coverage",
        "duplicate_targets",
        "wrong_commit",
        "duplicate_id",
        "namespace",
        "native_change",
        "symlink",
    ],
)
def test_unresolved_source_fails_before_publishing(tmp_path: Path, damage: str) -> None:
    root, best = _stack(tmp_path)
    first, second = best["optimization_stack"]
    path = Path(second["source_snapshot"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "missing_payload":
        (path.parent / "files/python/accepted_pkg/beta.py").unlink()
    elif damage == "wrong_payload":
        (path.parent / "files/python/accepted_pkg/beta.py").write_text("VALUE = 'corrupt'\n")
    elif damage == "missing_manifest":
        path.unlink()
    elif damage == "false_complete":
        second["source_snapshot_complete"] = False
    elif damage == "malformed_complete":
        manifest["complete"] = "false"
    elif damage == "missing_complete":
        manifest.pop("complete")
    elif damage == "malformed_schema":
        manifest["schema_version"] = True
    elif damage == "missing_coverage":
        manifest["extra"].pop("artifacts_outside_root")
    elif damage == "outside_artifact":
        manifest["extra"]["artifacts_outside_root"] = 1
    elif damage == "malformed_stack_coverage":
        second["source_artifacts_outside_root"] = False
    elif damage == "duplicate_targets":
        second["target_files"].append(second["target_files"][0])
    elif damage == "wrong_commit":
        second["base_sha"] = first["base_sha"]
        manifest["base_sha"] = first["base_sha"]
    elif damage == "duplicate_id":
        second["variant_name"] = first["variant_name"]
    elif damage == "namespace":
        best["optimization_stack"].append(
            _keep(root, tmp_path / "third", {"python/namespace_pkg/new.py": "VALUE = 1\n"})
        )
    elif damage == "native_change":
        best["optimization_stack"].append(
            _keep(root, tmp_path / "third", {"python/accepted_pkg/kernel.cpp": "int x = 1;\n"})
        )
    elif damage == "symlink":
        (root / "python/accepted_pkg/link").symlink_to("alpha.py")
        best["optimization_stack"].append(
            _keep(root, tmp_path / "third", {"python/accepted_pkg/alpha.py": "VALUE = 'third'\n"})
        )
    if damage != "missing_manifest":
        path.write_text(json.dumps(manifest))
    output = tmp_path / "materialized"
    with pytest.raises(SourceMaterializationError):
        materialize_source_stack(best, output)
    assert not output.exists() or not list(output.iterdir())


def test_multiple_trees_and_import_roots_preserve_order(tmp_path: Path) -> None:
    root, best = _stack(tmp_path)
    other = _repository(tmp_path / "other", package="other_pkg", prefix="src")
    best["optimization_stack"].insert(
        1, _keep(other, tmp_path / "other-layer", {"src/other_pkg/alpha.py": "VALUE = 'other'\n"}, prefix="src")
    )
    best["optimization_stack"].append(
        _keep(
            root,
            tmp_path / "third",
            {"lib/second_pkg/__init__.py": "", "lib/second_pkg/new.py": "VALUE = 'extra'\n"},
            prefix="lib",
        )
    )
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    manifest = json.loads((Path(descriptor["bundle_root"]) / "manifest.json").read_bytes())
    assert manifest["required_layer_ids"] == ["first", "other-layer", "second", "third"]
    assert [tree["layer_ids"] for tree in manifest["trees"]] == [["first", "second", "third"], ["other-layer"]]
    assert descriptor["pythonpath_prefixes"] == ["trees/000/python", "trees/000/lib", "trees/001/src"]


def test_corrupt_materialization_is_not_reused(tmp_path: Path) -> None:
    _, best = _stack(tmp_path)
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    target = Path(descriptor["bundle_root"]) / "trees/000/python/accepted_pkg/alpha.py"
    target.write_text("VALUE = 'corrupt'\n")
    with pytest.raises(SourceMaterializationError, match="corrupt_materialization"):
        materialize_source_stack(best, tmp_path / "materialized")


def test_repo_root_imports_allow_ancillary_source_and_ignore_archive_attributes(tmp_path: Path) -> None:
    root = _repository(tmp_path / "repo", prefix="")
    (root / "setup.py").write_text("raise RuntimeError('ancillary file must not execute')\n")
    (root / "tests").mkdir()
    (root / "tests/test_unused.py").write_text("raise RuntimeError('ancillary file must not execute')\n")
    (root / ".gitattributes").write_text("accepted_pkg/alpha.py export-ignore\n")
    layer = _keep(root, tmp_path / "first", {"accepted_pkg/alpha.py": "VALUE = 'first'\n"}, prefix="")
    descriptor = materialize_source_stack({"optimization_stack": [layer]}, tmp_path / "materialized")
    result = _imports(descriptor, tmp_path / "missing", tmp_path)
    assert result["alpha"]["value"] == "first"
    assert (Path(descriptor["bundle_root"]) / "trees/000/setup.py").is_file()


@pytest.mark.parametrize("damage", ["directory", "symlink", "mode"])
def test_cache_rejects_directory_symlink_and_mode_drift(tmp_path: Path, damage: str) -> None:
    _, best = _stack(tmp_path)
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    package = Path(descriptor["bundle_root"]) / "trees/000/python/accepted_pkg"
    if damage == "directory":
        (package / "removed").mkdir()
    elif damage == "symlink":
        (package / "unlisted").symlink_to(".", target_is_directory=True)
    else:
        (package / "alpha.py").chmod(0o755)
    with pytest.raises(SourceMaterializationError, match="corrupt_materialization"):
        materialize_source_stack(best, tmp_path / "materialized")


def test_legacy_explicit_coverage_is_preserved_without_inventing_it(tmp_path: Path) -> None:
    from hyperloom.orchestrator.loop.writeback import _source_layer_handles

    _, best = _stack(tmp_path)
    for entry in best["optimization_stack"]:
        path = Path(entry["source_snapshot"]) / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["extra"].pop("artifacts_outside_root")
        path.write_text(json.dumps(manifest))
        entry.update(_source_layer_handles({**entry, "source_artifacts_outside_root": 0}))
    assert materialize_source_stack(best, tmp_path / "materialized")["status"] == "ready"
    assert "source_artifacts_outside_root" not in _source_layer_handles({})


def test_competing_package_roots_and_native_artifacts_are_refused(tmp_path: Path) -> None:
    root, best = _stack(tmp_path)
    other = _repository(tmp_path / "other")
    best["optimization_stack"].append(
        _keep(other, tmp_path / "other-layer", {"python/accepted_pkg/alpha.py": "VALUE = 'other'\n"})
    )
    with pytest.raises(SourceMaterializationError, match="ambiguous_import_roots"):
        materialize_source_stack(best, tmp_path / "materialized")
    best["optimization_stack"].pop()
    native_root = _repository(tmp_path / "native")
    (native_root / "python/accepted_pkg/native.so").write_bytes(b"compiled-artifact")
    best["optimization_stack"] = [
        _keep(native_root, tmp_path / "third", {"python/accepted_pkg/alpha.py": "VALUE = 'third'\n"})
    ]
    with pytest.raises(SourceMaterializationError, match="unsupported_native_runtime"):
        materialize_source_stack(best, tmp_path / "materialized")


@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize("filename", ["native.SO.1", "generated.py", "config.json", "sourceless.pyc"])
def test_untracked_runtime_is_refused_but_python_cache_is_allowed(tmp_path: Path, ignored: bool, filename: str) -> None:
    root, best = _stack(tmp_path)
    package = root / "python/accepted_pkg"
    if ignored:
        (root / ".gitignore").write_text(filename + "\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/alpha.cpython-310.pyc").write_bytes(b"cache")
    assert materialize_source_stack(best, tmp_path / "materialized")["status"] == "ready"
    (package / filename).write_bytes(b"runtime-build-product")
    with pytest.raises(SourceMaterializationError, match="unsupported_untracked_runtime"):
        materialize_source_stack(best, tmp_path / "materialized")


@pytest.mark.parametrize("relative", ["support/generated.py", "helper.py", "namespace/data.json"])
def test_untracked_sibling_dependencies_under_import_prefix_are_refused(tmp_path: Path, relative: str) -> None:
    root = _repository(tmp_path / "repo")
    (root / "python/support").mkdir()
    (root / "python/support/__init__.py").write_text("")
    layer = _keep(root, tmp_path / "first", {"python/accepted_pkg/alpha.py": "import support\nVALUE = 'first'\n"})
    target = root / "python" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("unversioned-runtime-content")
    with pytest.raises(SourceMaterializationError, match="unsupported_untracked_runtime"):
        materialize_source_stack({"optimization_stack": [layer]}, tmp_path / "materialized")


@pytest.mark.parametrize(
    "relative", ["sitecustomize.py", "sitecustomize/__init__.py", "sitecustomize.pyc", "usercustomize/__init__.py"]
)
def test_startup_hooks_are_refused_as_modules_packages_or_bytecode(tmp_path: Path, relative: str) -> None:
    root = _repository(tmp_path / "repo")
    hook = root / "python" / relative
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("raise RuntimeError('must not execute')\n")
    layer = _keep(root, tmp_path / "first", {"python/accepted_pkg/alpha.py": "VALUE = 'first'\n"})
    with pytest.raises(SourceMaterializationError, match="unsupported_startup_hook"):
        materialize_source_stack({"optimization_stack": [layer]}, tmp_path / "materialized")


@pytest.mark.parametrize("name", ["alpha", "removed"])
def test_module_package_collisions_cannot_mask_changes_or_deletions(tmp_path: Path, name: str) -> None:
    root, best = _stack(tmp_path)
    best["optimization_stack"].append(
        _keep(root, tmp_path / "third", {f"python/accepted_pkg/{name}/__init__.py": "VALUE = 'package'\n"})
    )
    with pytest.raises(SourceMaterializationError, match="ambiguous_module_ownership"):
        materialize_source_stack(best, tmp_path / "materialized")


@pytest.mark.parametrize("path", ["alpha.py", "unchanged.py"])
def test_unrecorded_overwrites_and_sibling_changes_between_layers_are_refused(tmp_path: Path, path: str) -> None:
    root, best = _stack(tmp_path)
    (root / "python/accepted_pkg" / path).write_text("VALUE = 'unrecorded'\n")
    best["optimization_stack"].append(
        _keep(root, tmp_path / "third", {"python/accepted_pkg/beta.py": "VALUE = 'third'\n"})
    )
    with pytest.raises(SourceMaterializationError, match="unrecorded_source_changes"):
        materialize_source_stack(best, tmp_path / "materialized")


@pytest.mark.parametrize("dependency", ["support.py", "support/__init__.py", "support/namespace.py"])
def test_untouched_dependency_names_cannot_collide_across_import_roots(tmp_path: Path, dependency: str) -> None:
    layers = []
    for name in ("first", "second"):
        root = _repository(tmp_path / name, package=f"{name}_pkg")
        target = root / "python" / dependency
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = {name!r}\n")
        layers.append(
            _keep(
                root,
                tmp_path / f"{name}-snapshot",
                {f"python/{name}_pkg/alpha.py": "import support\nVALUE = support.VALUE\n"},
            )
        )
    with pytest.raises(SourceMaterializationError, match="ambiguous_import_roots"):
        materialize_source_stack({"optimization_stack": layers}, tmp_path / "materialized")


@pytest.mark.asyncio
@pytest.mark.parametrize("source_state", ["ready", "unresolved", "absent", "changed"])
async def test_kernel_handoff_prepares_source_or_stops_before_delegation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_state: str
) -> None:
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.shared_state import SharedState

    _, best = _stack(tmp_path)
    if source_state == "unresolved":
        best["optimization_stack"][-1]["source_snapshot"] = ""
    elif source_state == "absent":
        best = {}
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        current_best=best, baseline_tput=100.0, model_path="/models/fixture", gpu_type="mi355x", isl=1, osl=1, conc=1
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None

    def prepare_source(snapshot: dict, output: Path) -> dict:
        assert source_state != "absent", "no-source handoffs must not assemble source"
        source = materialize_source_stack(snapshot, output)
        if source_state == "changed":
            coord.shared_state.current_best["extra_server_args"] = "--changed-during-assembly"
        return source

    monkeypatch.setattr("hyperloom.orchestrator.source_materialization.materialize_source_stack", prepare_source)

    def stop_after_handoff(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kernel_agent_tool._kernel_agent_tool_path", stop_after_handoff
    )
    await coord._run_geak_kernel_phase(from_phase="KERNEL")
    handoff = tmp_path / "geak/handoff.json"
    if source_state in ("unresolved", "changed"):
        assert not handoff.exists()
        assert coord.shared_state.geak_result["error_class"] == "unresolved_baseline_source"
    else:
        spec = json.loads(handoff.read_text())["baseline_env_spec"]
        if source_state == "ready":
            assert spec["source_materialization"]["required_layer_ids"] == ["first", "second"]
        else:
            assert "source_materialization" not in spec


def test_no_source_is_unchanged_and_serializer_keeps_materialization_separate(tmp_path: Path) -> None:
    assert materialize_source_stack({}, tmp_path / "unused") is None
    assert not (tmp_path / "unused").exists()
    writer = object.__new__(WritebackCollaborator)
    writer.shared_state = SimpleNamespace(current_best={}, baseline_config_path="")
    original = writer.build_env_spec(server_launch_flags="")
    assert "source_materialization" not in original
    _, best = _stack(tmp_path)
    writer.shared_state.current_best = best
    descriptor = materialize_source_stack(best, tmp_path / "materialized")
    before = copy.deepcopy(descriptor)
    spec = writer.build_env_spec(server_launch_flags="", source_materialization=descriptor)
    assert spec["source_materialization"] == descriptor == before
    assert spec["launch_identity"] != writer.build_env_spec(server_launch_flags="")["launch_identity"]
    assert [row["id"] for row in spec["source_snapshots"]] == descriptor["required_layer_ids"]


@pytest.mark.parametrize(
    "declaration, supported", [("", False), ("= 1", True), ("= True", False), ("= 2", False), ("= int('1')", False)]
)
def test_capability_is_literal_and_does_not_execute_reader(tmp_path: Path, declaration: str, supported: bool) -> None:
    runner = tmp_path / "reader.py"
    runner.write_text(
        (f"SOURCE_MATERIALIZATION_SCHEMA_VERSION {declaration}\n" if declaration else "")
        + "raise RuntimeError('reader must not execute')\n"
    )
    assert geak_runner._supports_source_materialization(runner) is supported


def test_unsupported_consumer_is_refused_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = tmp_path / "reader.py"
    marker = tmp_path / "executed"
    runner.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    result = geak_runner.call_geak({"baseline_env_spec": {"source_materialization": {}}}, tmp_path / "run")
    assert result["error_class"] == "unsupported_source_materialization_reader"
    assert not marker.exists()


def test_legacy_source_handoff_cannot_execute_an_old_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = tmp_path / "reader.py"
    marker = tmp_path / "executed"
    runner.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    result = geak_runner.call_geak({"baseline_env_spec": {"source_snapshots": [{"id": "legacy"}]}}, tmp_path / "run")
    assert result["error_class"] == "unresolved_baseline_source"
    assert not marker.exists()


def test_controls_only_handoff_does_not_require_reader_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "reader.py"
    runner.write_text('import pathlib,sys\npathlib.Path(sys.argv[2]).write_text(\'{"status": "ok"}\')\n')
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    result = geak_runner.call_geak({"baseline_env_spec": {"source_snapshots": []}}, tmp_path / "run")
    assert result["status"] == "ok" and result["returncode"] == 0
