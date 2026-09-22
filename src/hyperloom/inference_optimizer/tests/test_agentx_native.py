# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native Magpie AgentX launcher validation tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import native as native_agentx
from hyperloom.inference_optimizer.agentx.native import (
    native_agentx_enabled,
    resolve_native_launcher,
    resolve_native_recipe,
    validate_native_launcher_name,
)


class _FakeDistribution:
    def __init__(self, direct_url: dict | None, package_root: Path):
        self._direct_url = direct_url
        self._package_root = package_root

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return json.dumps(self._direct_url) if self._direct_url is not None else None

    def locate_file(self, name: str) -> Path:
        assert name == "Magpie"
        return self._package_root


def _magpie_identity_namespace(
    direct_url: dict | None,
    *,
    distribution_root: Path | None = None,
) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(native_agentx._MAGPIE_SOURCE_IDENTITY_CODE, namespace)

    def _distribution(name: str):
        if name != "magpie-eval":
            pytest.fail(f"unexpected distribution: {name}")
        if distribution_root is None:
            raise namespace["PackageNotFoundError"]
        return _FakeDistribution(direct_url, distribution_root)

    namespace["distribution"] = _distribution
    return namespace


def _magpie_identity_resolver(
    direct_url: dict | None,
    *,
    distribution_root: Path | None = None,
):
    return _magpie_identity_namespace(
        direct_url,
        distribution_root=distribution_root,
    )["_resolve_magpie_source_identity"]


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Hyperloom test"], check=True)
    marker = path / "marker"
    marker.write_text("outer\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "marker"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_all(path: Path, message: str = "fixture update") -> str:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", message], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_magpie_wheel_identity_prefers_direct_url_over_unrelated_outer_git(tmp_path: Path):
    outer = tmp_path / "hyperloom"
    outer_commit = _init_git_repo(outer)
    package_root = outer / ".venv" / "lib" / "python3.12" / "site-packages" / "Magpie"
    package_root.mkdir(parents=True)
    pinned_magpie_commit = "a" * 40
    resolve = _magpie_identity_resolver(
        {
            "url": "https://github.com/AMD-AGI/Magpie.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": pinned_magpie_commit,
                "requested_revision": pinned_magpie_commit,
            },
        },
        distribution_root=package_root,
    )

    commit, url = resolve(package_root)

    assert commit == pinned_magpie_commit
    assert commit != outer_commit
    assert url == "https://github.com/AMD-AGI/Magpie.git"


def test_magpie_wheel_without_vcs_provenance_does_not_claim_outer_git(tmp_path: Path):
    outer = tmp_path / "hyperloom"
    _init_git_repo(outer)
    package_root = outer / ".venv" / "lib" / "python3.12" / "site-packages" / "Magpie"
    package_root.mkdir(parents=True)
    resolve = _magpie_identity_resolver(None, distribution_root=package_root)

    assert resolve(package_root) == ("", "")


def test_magpie_source_checkout_uses_its_own_git_top_level(tmp_path: Path):
    checkout = tmp_path / "Magpie-source"
    _init_git_repo(checkout)
    package_root = checkout / "Magpie"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    expected = _commit_all(checkout)
    resolve = _magpie_identity_resolver(None)

    assert resolve(package_root) == (expected, checkout.resolve().as_uri())


def test_magpie_editable_install_validates_declared_source_checkout(tmp_path: Path):
    checkout = tmp_path / "Magpie-editable"
    _init_git_repo(checkout)
    package_root = checkout / "Magpie"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    expected = _commit_all(checkout)
    resolve = _magpie_identity_resolver(
        {"url": checkout.resolve().as_uri(), "dir_info": {"editable": True}},
        distribution_root=package_root,
    )

    assert resolve(package_root) == (expected, checkout.resolve().as_uri())


def test_magpie_wheel_identity_rejects_distribution_import_mismatch(tmp_path: Path):
    imported_root = tmp_path / "imported" / "Magpie"
    imported_root.mkdir(parents=True)
    distribution_root = tmp_path / "installed" / "Magpie"
    distribution_root.mkdir(parents=True)
    commit = "a" * 40
    resolve = _magpie_identity_resolver(
        {
            "url": "https://github.com/AMD-AGI/Magpie.git",
            "vcs_info": {"vcs": "git", "commit_id": commit},
        },
        distribution_root=distribution_root,
    )

    with pytest.raises(RuntimeError, match="does not own the imported Magpie"):
        resolve(imported_root)


def test_magpie_source_identity_rejects_dirty_package_tree(tmp_path: Path):
    checkout = tmp_path / "Magpie-source"
    _init_git_repo(checkout)
    package_root = checkout / "Magpie"
    package_root.mkdir()
    module = package_root / "__init__.py"
    module.write_text("# fixture\n", encoding="utf-8")
    _commit_all(checkout)
    module.write_text("# tampered\n", encoding="utf-8")
    resolve = _magpie_identity_resolver(None)

    with pytest.raises(RuntimeError, match="clean Magpie/ source tree"):
        resolve(package_root)


def test_magpie_source_identity_ignores_generated_pycache(tmp_path: Path):
    checkout = tmp_path / "Magpie-source"
    _init_git_repo(checkout)
    package_root = checkout / "Magpie"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    expected = _commit_all(checkout)
    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-313.pyc").write_bytes(b"generated")
    resolve = _magpie_identity_resolver(None)

    assert resolve(package_root) == (expected, checkout.resolve().as_uri())


def test_magpie_published_execution_tree_pin_and_single_file_tamper(tmp_path: Path):
    namespace = _magpie_identity_namespace(None)
    package_root = tmp_path / "Magpie"
    (package_root / "nested").mkdir(parents=True)
    (package_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    script = package_root / "nested" / "launch.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    (package_root / "config.yaml").write_text("enabled: true\n", encoding="utf-8")
    (package_root / "mcp").mkdir()
    (package_root / "mcp" / "config.json").write_text("{}\n", encoding="utf-8")
    (package_root / "__pycache__").mkdir()
    (package_root / "__pycache__" / "ignored.py").write_text("tamper = True\n", encoding="utf-8")

    identify = namespace["_magpie_execution_tree_identity"]
    validate = namespace["_validate_magpie_execution_tree"]
    commit = "f" * 40
    pristine = identify(package_root)
    namespace["_AUDITED_MAGPIE_EXECUTION_TREES"][commit] = pristine

    assert validate(package_root, commit) == pristine
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="published execution tree differs"):
        validate(package_root, commit)


def test_magpie_pinned_published_tree_hash_matches_independent_wheel_audit():
    namespace = _magpie_identity_namespace(None)

    assert namespace["_AUDITED_MAGPIE_EXECUTION_TREES"] == {
        "3642ce66ae46ca4dc125340b3d14a3f4640c369b": {
            "file_count": 77,
            "tree_sha256": ("af54be3412932f6ac3556bf2eb498c860785b722ef5c0f2539d26f3a9f4ee204"),
        }
    }
    assert namespace["_MAGPIE_UNPUBLISHED_PATHS"] == frozenset({"mcp/config.json"})


def test_resolver_and_preflight_health_share_magpie_identity_guard():
    from hyperloom.inference_optimizer.cli import preflight

    guard = native_agentx._MAGPIE_SOURCE_IDENTITY_CODE
    assert native_agentx._RESOLVER_CODE.startswith(guard)
    assert preflight._MAGPIE_NATIVE_AGENTX_HEALTH_CODE.startswith(guard)
    assert preflight._magpie_health_code(native_agentx=True).startswith(guard)


@pytest.mark.parametrize(
    "custom_ref",
    ["v0.2.0", "release/custom-magpie", "f" * 40],
)
def test_generic_magpie_health_keeps_custom_refs_on_importability_contract(
    custom_ref: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """A generic Magpie ref must not be rejected by native-only provenance policy."""
    from hyperloom.inference_optimizer.cli import preflight

    code = preflight._magpie_health_code(native_agentx=False)

    assert code == preflight._MAGPIE_GENERIC_HEALTH_CODE
    assert code.strip() == "import Magpie"
    assert "_validate_magpie_execution_tree" not in code
    assert "AgentXConfig" not in code
    monkeypatch.setitem(sys.modules, "Magpie", ModuleType("Magpie"))
    monkeypatch.setattr(sys, "argv", ["magpie-health", custom_ref])
    exec(code, {})


def test_native_launch_config_hash_binds_forwarded_environment() -> None:
    base = {
        "framework": "sglang",
        "envs": {"CONC": 8, "PYTHONPATH": "/candidate/a"},
        "agentx": {"enabled": True},
    }
    reordered = {
        "agentx": {"enabled": True},
        "envs": {"PYTHONPATH": "/candidate/a", "CONC": 8},
        "framework": "sglang",
    }
    changed = {**base, "envs": {**base["envs"], "PYTHONPATH": "/candidate/b"}}

    assert native_agentx._canonical_benchmark_sha256(base) == native_agentx._canonical_benchmark_sha256(reordered)
    assert native_agentx._canonical_benchmark_sha256(base) != native_agentx._canonical_benchmark_sha256(changed)


@pytest.mark.parametrize("value", [True, "enable", "enabled", {"enabled": True}])
def test_native_agentx_enabled_values(value):
    assert native_agentx_enabled(value) is True


@pytest.mark.parametrize("value", [False, "disabled", None, {"enabled": False}])
def test_native_agentx_disabled_values(value):
    assert native_agentx_enabled(value) is False


def test_validate_native_launcher_accepts_explicit_agentic_script():
    path = validate_native_launcher_name("single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh")
    assert path.parts == ("single_node", "agentic", "dsv4_fp4_mi355x_sglang_mtp.sh")


@pytest.mark.parametrize(
    "script",
    ["", "/tmp/run.sh", "../run.sh", "sglang_mi355x.sh", "single_node/fixed_seq_len/run.sh"],
)
def test_validate_native_launcher_rejects_non_agentic_paths(script):
    with pytest.raises(ValueError, match="single_node/agentic"):
        validate_native_launcher_name(script)


def test_resolve_native_launcher_requires_existing_file(tmp_path: Path):
    rel = "single_node/agentic/run.sh"
    launcher = tmp_path / "benchmarks" / rel
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    assert resolve_native_launcher(inferencex_path=tmp_path, benchmark_script=rel) == launcher


def test_resolve_native_launcher_does_not_modify_upstream(tmp_path: Path):
    rel = "single_node/agentic/run.sh"
    launcher = tmp_path / "benchmarks" / rel
    launcher.parent.mkdir(parents=True)
    original = "#!/usr/bin/env bash\nSGLANG_CMD=(python3 -m sglang.launch_server)\n"
    launcher.write_text(original, encoding="utf-8")

    resolve_native_launcher(inferencex_path=tmp_path, benchmark_script=rel)

    assert launcher.read_text(encoding="utf-8") == original


def test_resolve_native_launcher_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_native_launcher(
            inferencex_path=tmp_path,
            benchmark_script="single_node/agentic/missing.sh",
        )


@pytest.mark.parametrize(
    "suffix",
    ["Inference X", "InferenceX;touch-pwned", "InferenceX$(id)", "InferenceX\nnext"],
)
def test_native_boundaries_reject_shell_unsafe_checkout_paths(tmp_path: Path, suffix: str):
    unsafe = tmp_path / suffix

    with pytest.raises(ValueError, match="shell-unsafe"):
        resolve_native_launcher(
            inferencex_path=unsafe,
            benchmark_script="single_node/agentic/run.sh",
        )
    with pytest.raises(ValueError, match="shell-unsafe"):
        native_agentx.preview_native_recipe(
            {"framework": "sglang", "agentx": {"enabled": True}},
            inferencex_path=unsafe,
        )


def test_native_launch_environment_identity_binds_effective_audited_controls() -> None:
    benchmark = {"envs": {"SPEC_NUM_TOKENS": 2}}
    base_env = {
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "SCHEDULER_RECV_INTERVAL": "30",
        "SPEC_NUM_TOKENS": "99",
        "HICACHE_RATIO": "1.5",
        "PYTHONHOME": "/opt/python-a",
        "HF_TOKEN": "secret-a",
        "HF_HOME": "/cache/a",
        "LD_PRELOAD": "/tmp/blocked-a.so",
        "AIPERF_BIN": "/tmp/untrusted-aiperf",
    }
    original = native_agentx.native_launch_environment_identity(benchmark, launch_env=base_env)

    assert {
        "HICACHE_RATIO",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONHOME",
        "SCHEDULER_RECV_INTERVAL",
        "SPEC_NUM_TOKENS",
    }.issubset(original["names"])
    assert "HF_TOKEN" not in original["names"]
    assert "HF_HOME" not in original["names"]
    assert "LD_PRELOAD" not in original["names"]
    assert "AIPERF_BIN" not in original["names"]

    for name, changed_value in (
        ("SCHEDULER_RECV_INTERVAL", "31"),
        ("LD_LIBRARY_PATH", "/opt/rocm/other-lib"),
        ("HICACHE_RATIO", "0.5"),
        ("PYTHONHOME", "/opt/python-b"),
    ):
        changed_env = {**base_env, name: changed_value}
        changed = native_agentx.native_launch_environment_identity(benchmark, launch_env=changed_env)
        assert changed["sha256"] != original["sha256"]

    # Magpie's benchmark.envs overlay is authoritative, and generic child-env
    # scrubbing happens before identity selection.
    assert (
        native_agentx.native_launch_environment_identity(
            benchmark,
            launch_env={**base_env, "SPEC_NUM_TOKENS": "100"},
        )
        == original
    )
    assert (
        native_agentx.native_launch_environment_identity(
            benchmark,
            launch_env={**base_env, "HF_TOKEN": "secret-b", "LD_PRELOAD": "/tmp/blocked-b.so"},
        )
        == original
    )


def test_native_launch_environment_scrubs_exported_bash_functions() -> None:
    """An inherited function must not override commands in Magpie's ``bash -c`` launcher."""
    from hyperloom.common.env_safety import scrub_benchmark_process_env

    function_key = "BASH_FUNC_hyperloom_agentx_probe%%"
    clean_env = {"PATH": "/usr/bin:/bin"}
    poisoned_env = {
        **clean_env,
        function_key: "() { printf 'injected\\n'; }",
    }

    assert native_agentx.native_launch_environment_identity({}, launch_env=poisoned_env) == (
        native_agentx.native_launch_environment_identity({}, launch_env=clean_env)
    )

    child_env = scrub_benchmark_process_env(dict(poisoned_env))
    native_agentx.scrub_native_agentx_ambient_env(child_env)
    assert function_key not in child_env

    probe = subprocess.run(
        ["/bin/bash", "-c", "type hyperloom_agentx_probe"],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    assert "function" not in f"{probe.stdout}\n{probe.stderr}".lower()


@pytest.mark.parametrize(
    "name",
    [
        "SCHEDULER_RECV_INTERVAL",
        "SPEC_NUM_TOKENS",
        "LD_LIBRARY_PATH",
        "GPU_MEM_UTIL",
        "MAX_NUM_SEQS",
        "MAX_CUDAGRAPH_CAPTURE_SIZE",
        "SYNTHETIC_ACCEPT_LEN",
        "L3_PER_RANK_GB",
        "HICACHE_RATIO",
        "HICACHE_WRITE_POLICY",
        "HICACHE_IO_BACKEND",
        "HICACHE_MEM_LAYOUT",
    ],
)
def test_audited_launcher_ambient_knobs_are_execution_controls(name: str) -> None:
    baseline = native_agentx.native_launch_environment_identity({}, launch_env={})
    changed = native_agentx.native_launch_environment_identity({}, launch_env={name: "test-value"})

    assert name in changed["names"]
    assert changed["sha256"] != baseline["sha256"]


def _make_inferencex_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    aiperf_source = tmp_path / "aiperf-source"
    _init_git_repo(aiperf_source)
    (aiperf_source / "pyproject.toml").write_text(
        "[project]\nname = 'aiperf'\nversion = '0'\n",
        encoding="utf-8",
    )
    _commit_all(aiperf_source, "add package")

    checkout = tmp_path / "InferenceX"
    _init_git_repo(checkout)
    launcher_rel = "single_node/agentic/run.sh"
    launcher = checkout / "benchmarks" / launcher_rel
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    (checkout / "benchmarks" / "benchmark_lib.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    configs = checkout / "configs"
    configs.mkdir()
    (configs / "amd-master.yaml").write_text("models: []\n", encoding="utf-8")
    (configs / "runners.yaml").write_text("runners: {}\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(aiperf_source),
            "utils/aiperf",
        ],
        check=True,
    )
    head = _commit_all(checkout, "add native inputs")
    return checkout, head, launcher_rel


def test_native_execution_identity_binds_and_rejects_runners_yaml(
    tmp_path: Path,
    monkeypatch,
):
    checkout, head, launcher_rel = _make_inferencex_checkout(tmp_path)
    launcher_sha256 = native_agentx._sha256_file(checkout / "benchmarks" / launcher_rel)
    monkeypatch.setattr(
        native_agentx,
        "_NATIVE_LAUNCHER_MANIFEST",
        {(head, "fixture-recipe"): (launcher_rel, launcher_sha256)},
    )
    magpie_commit = "a" * 40
    magpie_execution = {
        "fingerprint": "b" * 64,
        "source_commit": magpie_commit,
    }
    resolved_benchmark = {"framework": "sglang", "envs": {"TP": 1}}

    identity = native_agentx.native_execution_identity(
        inferencex_path=checkout,
        benchmark_script=launcher_rel,
        config_file="configs/amd-master.yaml",
        resolved_benchmark=resolved_benchmark,
        expected_ref=head,
        magpie_execution=magpie_execution,
        expected_magpie_ref=magpie_commit,
        launch_env={"SCHEDULER_RECV_INTERVAL": "30", "LD_LIBRARY_PATH": "/opt/rocm/lib"},
    )
    changed_ambient = native_agentx.native_execution_identity(
        inferencex_path=checkout,
        benchmark_script=launcher_rel,
        config_file="configs/amd-master.yaml",
        resolved_benchmark=resolved_benchmark,
        expected_ref=head,
        magpie_execution=magpie_execution,
        expected_magpie_ref=magpie_commit,
        launch_env={"SCHEDULER_RECV_INTERVAL": "31", "LD_LIBRARY_PATH": "/opt/rocm/lib"},
    )

    runners = checkout / "configs" / "runners.yaml"
    assert identity["runners_config_sha256"] == native_agentx._sha256_file(runners)
    assert identity["static_execution_fingerprint"] == changed_ambient["static_execution_fingerprint"]
    assert identity["launch_environment"] != changed_ambient["launch_environment"]
    assert identity["execution_fingerprint"] != changed_ambient["execution_fingerprint"]
    runners.write_text("runners:\n  tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean pinned launcher/replay inputs"):
        native_agentx.native_execution_identity(
            inferencex_path=checkout,
            benchmark_script=launcher_rel,
            config_file="configs/amd-master.yaml",
            resolved_benchmark=resolved_benchmark,
            expected_ref=head,
            magpie_execution=magpie_execution,
            expected_magpie_ref=magpie_commit,
        )


def test_recipe_resolution_uses_benchmark_interpreter(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import benchmark_backend

    seen = {}
    monkeypatch.setattr(
        benchmark_backend,
        "resolve_benchmark_interpreter",
        lambda: "/opt/benchmark/bin/python",
    )

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["payload"] = json.loads(kwargs["input"])
        result = {
            "benchmark": {"framework": "sglang", "envs": {}},
            "recipe": "recipe-a",
            "config_file": "configs/amd-master.yaml",
            "entry": {"tp": 1},
        }
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=native_agentx._RESOLVER_SENTINEL + json.dumps(result) + "\n",
            stderr="",
        )

    monkeypatch.setattr(native_agentx.subprocess, "run", _run)
    result = native_agentx._run_magpie_recipe_resolver(
        {"framework": "sglang"},
        inferencex_path=tmp_path,
    )

    assert seen["cmd"][0] == "/opt/benchmark/bin/python"
    assert seen["payload"]["inferencex_path"] == str(tmp_path)
    assert result["recipe"] == "recipe-a"


def _install_fake_magpie(monkeypatch, *, entry: dict):
    def _resolve(benchmark, *, inferencex_path):
        assert inferencex_path.is_dir()
        result = dict(benchmark)
        envs = dict(result.get("envs") or {})
        envs.update(
            {
                "TP": entry["tp"],
                "PP_SIZE": entry.get("pp", 1),
                "PCP_SIZE": entry.get("pcp-size", 1),
                "RECIPE_FINGERPRINT": entry["recipe-fingerprint"],
            }
        )
        result["envs"] = envs
        result["docker_image"] = entry["image"]
        result.setdefault("gpu_selection", {"auto": True})
        return {
            "benchmark": result,
            "recipe": "recipe-a",
            "config_file": "configs/amd-master.yaml",
            "entry": dict(entry),
        }

    monkeypatch.setattr(native_agentx, "_run_magpie_recipe_resolver", _resolve)
    monkeypatch.setattr(
        native_agentx,
        "validate_native_recipe_launcher",
        lambda **_kwargs: {
            "inferencex_commit": "1" * 40,
            "recipe": "recipe-a",
            "launcher": "single_node/agentic/run.sh",
            "launcher_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        native_agentx,
        "native_execution_identity",
        lambda **_kwargs: {
            "static_execution_fingerprint": "3" * 64,
            "execution_fingerprint": "3" * 64,
            "inferencex_commit": "1" * 40,
            "magpie_commit": "4" * 40,
        },
    )


def _recipe_entry(**overrides):
    value = {
        "tp": 4,
        "pp": 1,
        "pcp-size": 1,
        "ep": 4,
        "conc": 8,
        "duration": 3600,
        "image": "example/sglang:agentx",
        "runner": "cluster:mi355x-amds",
        "recipe-fingerprint": "a" * 64,
    }
    value.update(overrides)
    return value


def test_resolve_native_recipe_persists_topology_and_provenance(tmp_path, monkeypatch):
    entry = _recipe_entry()
    _install_fake_magpie(monkeypatch, entry=entry)
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
        "gpu_selection": {"auto": True},
        "workload_spec": {"kind": "agentx_trace_replay"},
    }

    topology = resolve_native_recipe(
        benchmark,
        inferencex_path=tmp_path,
        expected_gpu_count=4,
        outer_image=entry["image"],
    )

    assert topology == {
        "tp": 4,
        "pp": 1,
        "pcp_size": 1,
        "ep": 4,
        "gpu_count": 4,
        "conc": 8,
        "duration_seconds": 3600,
        "recipe_fingerprint": "a" * 64,
    }
    assert benchmark["envs"]["RECIPE_FINGERPRINT"] == "a" * 64
    assert benchmark["workload_spec"]["recipe"]["name"] == "recipe-a"
    assert benchmark["workload_spec"]["outer_image_config_pinned"] is True


def test_resolve_native_recipe_rejects_recipe_fingerprint_drift(
    tmp_path,
    monkeypatch,
):
    _install_fake_magpie(monkeypatch, entry=_recipe_entry())
    monkeypatch.setenv(
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT",
        "b" * 64,
    )
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
    }

    with pytest.raises(ValueError, match="recipe changed after session finalization"):
        resolve_native_recipe(
            benchmark,
            inferencex_path=tmp_path,
            expected_gpu_count=4,
            outer_image="example/sglang:agentx",
        )


def test_resolve_native_recipe_rejects_outer_topology_mismatch(tmp_path, monkeypatch):
    _install_fake_magpie(monkeypatch, entry=_recipe_entry(tp=8))
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
    }

    with pytest.raises(ValueError, match="--tp"):
        resolve_native_recipe(
            benchmark,
            inferencex_path=tmp_path,
            expected_gpu_count=4,
            outer_image="example/sglang:agentx",
        )


@pytest.mark.parametrize("outer_image", [None, "wrong/image:tag"])
def test_resolve_native_recipe_requires_matching_outer_image(
    tmp_path,
    monkeypatch,
    outer_image,
):
    _install_fake_magpie(monkeypatch, entry=_recipe_entry())
    monkeypatch.delenv("HYPERLOOM_IMAGE", raising=False)
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
    }

    with pytest.raises(ValueError, match="outer image"):
        resolve_native_recipe(
            benchmark,
            inferencex_path=tmp_path,
            expected_gpu_count=4,
            outer_image=outer_image,
        )


def test_resolve_native_recipe_honors_exact_manual_gpu_mask(tmp_path, monkeypatch):
    _install_fake_magpie(monkeypatch, entry=_recipe_entry())
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "0,1,2,3"},
        "gpu_selection": {"auto": True},
    }

    resolve_native_recipe(
        benchmark,
        inferencex_path=tmp_path,
        expected_gpu_count=4,
        outer_image="example/sglang:agentx",
    )

    assert benchmark["gpu_selection"]["auto"] is False


def test_resolve_native_recipe_rejects_nonzero_physical_mask(tmp_path, monkeypatch):
    _install_fake_magpie(monkeypatch, entry=_recipe_entry())
    benchmark = {
        "framework": "sglang",
        "model": "amd/model",
        "run_mode": "local",
        "runner_type": "mi355x",
        "envs": {"CONC": 8, "ROCR_VISIBLE_DEVICES": "4,5,6,7"},
    }

    with pytest.raises(ValueError, match="must be exactly"):
        resolve_native_recipe(
            benchmark,
            inferencex_path=tmp_path,
            expected_gpu_count=4,
            outer_image="example/sglang:agentx",
        )


def _native_grid_base(tmp_path: Path) -> Path:
    base = tmp_path / "native.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "amd/model",
                    "precision": "fp4",
                    "run_mode": "local",
                    "runner_type": "mi355x",
                    "agentx": {
                        "enabled": True,
                        "recipe": "recipe-a",
                        "resolved": {"tp": 4, "ep": 4, "conc": 8},
                    },
                    "benchmark_script": "single_node/agentic/run.sh",
                    "inferencex_path": str(tmp_path),
                    "envs": {
                        "AGENTX_MODEL_ID": "amd/model",
                        "AGENTX_SERVER_SCRIPT": "single_node/agentic/run.sh",
                        "CONC": 8,
                        "ROCR_VISIBLE_DEVICES": "0,1,2,3",
                        "TP": 4,
                    },
                    "workload_spec": {
                        "kind": "agentx_trace_replay",
                        "outer_image": "example/sglang:agentx",
                        "resolved_topology": {
                            "tp": 4,
                            "pp": 1,
                            "pcp_size": 1,
                            "ep": 4,
                            "gpu_count": 4,
                            "conc": 8,
                            "duration_seconds": 3600,
                            "recipe_fingerprint": "a" * 64,
                        },
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return base


@pytest.mark.parametrize(
    ("variant_kwargs", "build_kwargs", "match"),
    [
        ({"extra_server_args": "--foo 1"}, {}, "variant.extra_server_args"),
        ({"extra_envs": {"USE_AITER": "1"}}, {}, "extra_envs=USE_AITER"),
        ({"remove_args": ["--foo"]}, {}, "remove_args"),
        ({}, {"base_extra_args": "--bar 2"}, "base_extra_args"),
    ],
)
def test_native_grid_variant_rejects_unapplied_server_candidates(
    tmp_path,
    variant_kwargs,
    build_kwargs,
    match,
):
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    base = _native_grid_base(tmp_path)
    variant = GridVariant(name="candidate", **variant_kwargs)
    with pytest.raises(ValueError, match=match):
        _build_variant_yaml(
            base,
            build_kwargs.get("base_extra_args", ""),
            variant,
            output_subdir=tmp_path / "out",
        )


def test_native_grid_rejects_concurrency_change_before_recipe_resolution(
    tmp_path,
    monkeypatch,
):
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    base = _native_grid_base(tmp_path)
    monkeypatch.setattr(
        native_agentx,
        "resolve_native_recipe",
        lambda *_args, **_kwargs: pytest.fail("a fixed-CONC mismatch must fail before recipe resolution"),
    )

    with pytest.raises(ValueError, match="concurrency is fixed"):
        _build_variant_yaml(
            base,
            "",
            GridVariant(name="conc16", extra_envs={"CONC": "16"}),
            output_subdir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"tp": 2, "pp": 2},
        {"recipe_fingerprint": "b" * 64},
        {"ep": 2},
    ],
)
def test_native_grid_rejects_same_size_recipe_arm_changes(
    tmp_path,
    monkeypatch,
    changed,
):
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    topology = {
        "tp": 4,
        "pp": 1,
        "pcp_size": 1,
        "ep": 4,
        "gpu_count": 4,
        "conc": 8,
        "duration_seconds": 3600,
        "recipe_fingerprint": "a" * 64,
    }
    topology.update(changed)
    monkeypatch.setattr(
        native_agentx,
        "resolve_native_recipe",
        lambda *_args, **_kwargs: dict(topology),
    )

    with pytest.raises(ValueError, match="topology-changing rounds"):
        _build_variant_yaml(
            _native_grid_base(tmp_path),
            "",
            GridVariant(name="same-conc"),
            output_subdir=tmp_path / "out",
        )
