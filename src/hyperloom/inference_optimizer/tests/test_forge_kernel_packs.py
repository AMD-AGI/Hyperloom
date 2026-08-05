# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KernelForge optimized-kernel pack tests: discovery, install, gating, wiring.

A pack under ``$FORGE_PATH/serving_patches/kernels/`` is a *generated kernel*,
not a patch, so three things stand between it and a served token: the manifest
has to declare a call site, a GPU preflight has to verify the shapes on this
machine, and only then may the framework source be patched. These tests pin
each hop plus the choke point in ``materialize_config_with_envs``, none of
which needs a GPU (the preflight subprocess is stubbed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.orchestrator.actions.executors import _forge_kernel_patcher as fkp
from hyperloom.orchestrator.actions.executors import _workload_envs
from hyperloom.orchestrator.actions.executors._workload_envs import (
    materialize_config_with_envs,
)

_PACKS_ENV = fkp.ENV_ENABLED_PACKS


def _write_pack(root: Path, name: str = "flydsl_softmax", *, manifest: dict | None = None) -> Path:
    """Materialize a minimal KernelForge-style pack under ``<root>/serving_patches``."""
    pack_dir = root / "serving_patches" / "kernels" / name
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "kernel.py").write_text("def build_softmax_module(m, n, dt):\n    return None\n")
    _write_versioned_patch(root, "vllm", "vllm_0_25_0", "p.patch")
    (root / "serving_patches" / "vllm" / "SUPPORTED_VERSIONS.txt").write_text("0.25.0\n")
    doc = (
        manifest
        if manifest is not None
        else {
            "schema_version": 1,
            "name": name,
            "op": "rowwise_softmax",
            "language": "flydsl",
            "module": "kernel.py",
            "builder": "build_softmax_module",
            "correctness": {"min_snr_db": 30.0},
            "performance": {"min_graph_speedup": 1.1},
            "probe_shapes": [{"M": 4096, "N": 1024, "dtype": "f32"}],
            "targets": [
                {
                    "framework": "vllm",
                    "versions": ["0.25"],
                    "patch_name": "p.patch",
                    "sentinel": {"file": "some/mod.py", "markers": ["marker_a", "marker_b"]},
                }
            ],
        }
    )
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(doc))
    return pack_dir


def _write_versioned_patch(root: Path, framework: str, subdir: str, patch_name: str) -> Path:
    """Materialize one patch in the KernelForge versioned tree for ``framework``."""
    patch_dir = root / "serving_patches" / framework / subdir
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch = patch_dir / patch_name
    patch.write_text("diff --git a/x b/x\n")
    return patch


# ---------------------------------------------------------------- discovery


def test_discovers_pack_and_its_targets(tmp_path, monkeypatch):
    _write_pack(tmp_path)
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))

    packs = fkp.discover_packs()

    assert [p.name for p in packs] == ["flydsl_softmax"]
    target = packs[0].targets[0]
    assert (target.framework, target.versions) == ("vllm", ("0.25",))
    assert target.sentinel_markers == ("marker_a", "marker_b")


def test_bare_kernel_without_manifest_is_not_a_pack(tmp_path, monkeypatch):
    # Today's KernelForge output: kernel.py alone. There is no declared op or
    # call site, so Hyperloom must skip it rather than guess.
    wip = tmp_path / "serving_patches" / "kernels" / "wip"
    wip.mkdir(parents=True)
    (wip / "kernel.py").write_text("x = 1\n")
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))

    assert fkp.discover_packs() == ()


def test_discovery_is_empty_without_forge_path(monkeypatch):
    monkeypatch.delenv("FORGE_PATH", raising=False)
    assert fkp.discover_packs() == ()


# ------------------------------------------------------------------ install


def test_install_normalizes_manifest_to_json(tmp_path, monkeypatch):
    pack_dir = _write_pack(tmp_path)
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))
    pack = fkp._read_source_pack(pack_dir)

    installed = fkp._install(pack, tmp_path / "installed")

    assert (installed / "kernel.py").is_file()
    manifest = json.loads((installed / "pack.json").read_text())
    # Nested YAML sections are flattened so the serving process reads flat JSON
    # and never needs a YAML parser.
    assert manifest["op"] == "rowwise_softmax"
    assert manifest["min_snr_db"] == 30.0
    assert manifest["min_graph_speedup"] == 1.1
    assert len(manifest["kernel_sha256"]) == 64


def test_install_refreshes_a_changed_kernel(tmp_path, monkeypatch):
    pack_dir = _write_pack(tmp_path)
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))
    dest = tmp_path / "installed"
    first = json.loads((fkp._install(fkp._read_source_pack(pack_dir), dest) / "pack.json").read_text())

    (pack_dir / "kernel.py").write_text("def build_softmax_module(m, n, dt):\n    return 1\n")
    second = json.loads((fkp._install(fkp._read_source_pack(pack_dir), dest) / "pack.json").read_text())

    assert first["kernel_sha256"] != second["kernel_sha256"]


# ----------------------------------------------------------------- preflight


def test_preflight_reuses_a_report_for_the_same_kernel(tmp_path):
    installed = tmp_path / "pack"
    installed.mkdir()
    (installed / "preflight.json").write_text(
        json.dumps({"ok": True, "kernel_sha256": "abc", "verified": [{"N": 1024, "dtype": "f32"}]})
    )

    assert fkp._preflight(installed, {"name": "p", "kernel_sha256": "abc"}, force=False) is True


def test_preflight_rejects_a_previously_failed_kernel(tmp_path):
    installed = tmp_path / "pack"
    installed.mkdir()
    (installed / "preflight.json").write_text(
        json.dumps({"ok": False, "kernel_sha256": "abc", "reason": "every shape rejected"})
    )

    assert fkp._preflight(installed, {"name": "p", "kernel_sha256": "abc"}, force=False) is False


def test_preflight_reruns_when_the_kernel_changed(tmp_path, monkeypatch):
    installed = tmp_path / "pack"
    installed.mkdir()
    (installed / "preflight.json").write_text(json.dumps({"ok": True, "kernel_sha256": "old"}))
    spawned: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        spawned.append(cmd)
        (installed / "preflight.json").write_text(json.dumps({"ok": True, "verified": [{}]}))

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Proc()

    monkeypatch.setattr(fkp.subprocess, "run", _fake_run)

    assert fkp._preflight(installed, {"name": "p", "kernel_sha256": "new"}, force=False) is True
    assert len(spawned) == 1
    # The fresh verdict is stamped with the kernel it was produced from, so the
    # next run reuses it instead of re-gating.
    assert json.loads((installed / "preflight.json").read_text())["kernel_sha256"] == "new"


def test_preflight_failure_blocks_patching(tmp_path, monkeypatch):
    _write_pack(tmp_path)
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))
    monkeypatch.setattr(fkp, "_preflight", lambda *_a, **_k: False)
    patched: list[object] = []
    monkeypatch.setattr(fkp, "_ensure_patched", lambda plan: patched.append(plan) or True)

    landed = fkp.ensure_framework_patched_for_forge_kernels(
        "vllm", ["flydsl_softmax"], install_root=tmp_path / "installed"
    )

    assert landed == ()
    assert patched == []


# -------------------------------------------------------------- target match


@pytest.mark.parametrize(
    ("versions", "version", "expected"),
    [
        (["0.25"], "0.25.0", True),
        (["0.25"], "0.25.3", True),
        (["0.25"], "0.250.0", False),
        (["0.25"], "0.26.0", False),
        (["0.25.0"], "0.25.0", True),
        ([], "9.9.9", True),
    ],
)
def test_target_version_prefixes(tmp_path, versions, version, expected):
    pack_dir = _write_pack(tmp_path)
    doc = yaml.safe_load((pack_dir / "pack.yaml").read_text())
    doc["targets"][0]["versions"] = versions
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(doc))
    pack = fkp._read_source_pack(pack_dir)

    assert (fkp._select_target(pack, "vllm", version) is not None) is expected


def test_target_framework_must_match(tmp_path):
    pack = fkp._read_source_pack(_write_pack(tmp_path))
    assert fkp._select_target(pack, "sglang", "0.25.0") is None


# ------------------------------------------------------------ patch resolution


@pytest.fixture
def _no_version_pins(monkeypatch):
    for var in (
        "HYPERLOOM_VLLM_SERVING_PATCH_EXACT_VERSIONS",
        "HYPERLOOM_VLLM_SERVING_PATCH_ALLOWED_MINORS",
    ):
        monkeypatch.delenv(var, raising=False)


def _resolve(tmp_path: Path, version: str = "0.25.0"):
    pack = fkp._read_source_pack(tmp_path / "serving_patches" / "kernels" / "flydsl_softmax")
    target = fkp._select_target(pack, "vllm", version)
    return fkp._resolve_target_patch(pack, target, "vllm", version, tmp_path)


def test_patch_resolves_from_the_versioned_framework_tree(tmp_path, _no_version_pins):
    _write_pack(tmp_path)

    assert _resolve(tmp_path) == tmp_path / "serving_patches" / "vllm" / "vllm_0_25_0" / "p.patch"


def test_patch_falls_back_to_the_nearest_not_newer_version(tmp_path, _no_version_pins):
    _write_pack(tmp_path)
    (tmp_path / "serving_patches" / "vllm" / "SUPPORTED_VERSIONS.txt").write_text("0.25.0\n0.25.4\n")

    assert _resolve(tmp_path, "0.25.4") == (tmp_path / "serving_patches" / "vllm" / "vllm_0_25_0" / "p.patch")


def test_patch_skips_a_version_dir_without_the_named_patch(tmp_path, _no_version_pins):
    _write_pack(tmp_path)
    _write_versioned_patch(tmp_path, "vllm", "vllm_0_25_2", "unrelated.patch")
    (tmp_path / "serving_patches" / "vllm" / "SUPPORTED_VERSIONS.txt").write_text("0.25.0\n0.25.3\n")

    # 0.25.2 is nearer, but holds no p.patch, so 0.25.0 is the only candidate.
    assert _resolve(tmp_path, "0.25.3") == (tmp_path / "serving_patches" / "vllm" / "vllm_0_25_0" / "p.patch")


def test_unsupported_version_is_rejected_by_the_manifest(tmp_path, _no_version_pins):
    _write_pack(tmp_path)
    _write_versioned_patch(tmp_path, "vllm", "vllm_0_25_0", "p.patch")
    (tmp_path / "serving_patches" / "vllm" / "SUPPORTED_VERSIONS.txt").write_text("0.24.0\n")

    assert _resolve(tmp_path) is None


def test_missing_manifest_fails_closed_for_vllm(tmp_path, _no_version_pins):
    _write_pack(tmp_path)
    (tmp_path / "serving_patches" / "vllm" / "SUPPORTED_VERSIONS.txt").unlink()

    # vllm has no built-in minor allowlist, so an unvouched tree is not applied.
    assert _resolve(tmp_path) is None


# --------------------------------------------------------------- env helpers


def test_pack_envs_is_empty_when_nothing_landed():
    # An unconditional merge of this must never switch the feature on.
    assert fkp.pack_envs(()) == {}


def test_pack_envs_names_the_landed_packs(tmp_path):
    envs = fkp.pack_envs(("a", "b"), install_root=tmp_path)
    assert envs[_PACKS_ENV] == "a,b"
    assert envs[fkp.ENV_PACK_ROOT] == str(tmp_path)


def test_packs_requested_from_env_parses_csv():
    assert fkp.packs_requested_from_env({_PACKS_ENV: " a , b ,"}) == ("a", "b")
    assert fkp.packs_requested_from_env({}) == ()


# ------------------------------------------------- materialize_config wiring


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setattr(cli_model_gate, "_autodetect_gpu_type", lambda: None)
    for key in ("CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "PRECISION", "RUN_EVAL", "FRAMEWORK", _PACKS_ENV):
        monkeypatch.delenv(key, raising=False)


def _materialize(tmp_path: Path, *, framework: str = "vllm", extra_envs: dict | None = None) -> dict:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "/models/whatever",
                    "precision": "bf16",
                    "run_mode": "local",
                    "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
                    "timeout_seconds": 600,
                    "profiler": {
                        "torch_profiler": {"enabled": False},
                        "system_profiler": {"enabled": False},
                        "tracelens": {"enabled": False},
                    },
                    "gpu_selection": {"auto": False},
                }
            }
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base, out, model_path="/models/whatever", gpu_type=None, extra_envs=extra_envs
    )
    return yaml.safe_load(materialized.read_text())["benchmark"]["envs"]


def test_materialize_keeps_the_env_for_a_landed_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(_workload_envs, "ensure_framework_patched_for_forge_kernels", lambda *a, **k: ("p1",))
    monkeypatch.setattr(_workload_envs, "forge_pack_envs", lambda landed: {_PACKS_ENV: ",".join(landed)})

    envs = _materialize(tmp_path, extra_envs={_PACKS_ENV: "p1"})

    assert envs.get(_PACKS_ENV) == "p1"


def test_materialize_narrows_the_env_to_what_landed(tmp_path, monkeypatch):
    monkeypatch.setattr(_workload_envs, "ensure_framework_patched_for_forge_kernels", lambda *a, **k: ("p1",))
    monkeypatch.setattr(_workload_envs, "forge_pack_envs", lambda landed: {_PACKS_ENV: ",".join(landed)})

    envs = _materialize(tmp_path, extra_envs={_PACKS_ENV: "p1,p2"})

    assert envs.get(_PACKS_ENV) == "p1"


def test_materialize_drops_the_env_when_nothing_lands(tmp_path, monkeypatch):
    # Leaving the env set would make the benchmark record claim a kernel the
    # server is not running.
    monkeypatch.setattr(_workload_envs, "ensure_framework_patched_for_forge_kernels", lambda *a, **k: ())

    envs = _materialize(tmp_path, extra_envs={_PACKS_ENV: "p1"})

    assert _PACKS_ENV not in envs


def test_materialize_skips_the_patcher_without_the_env(tmp_path, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(
        _workload_envs,
        "ensure_framework_patched_for_forge_kernels",
        lambda *a, **k: calls.append(a) or (),
    )

    _materialize(tmp_path)

    assert calls == []


def test_materialize_honors_the_patch_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    calls: list[object] = []
    monkeypatch.setattr(
        _workload_envs,
        "ensure_framework_patched_for_forge_kernels",
        lambda *a, **k: calls.append(a) or (),
    )

    envs = _materialize(tmp_path, extra_envs={_PACKS_ENV: "p1"})

    assert calls == []
    # Kill switch means "do not touch the framework", not "rewrite the env".
    assert envs.get(_PACKS_ENV) == "p1"
