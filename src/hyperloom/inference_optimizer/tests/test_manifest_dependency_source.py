# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session dependencies name the checkout selected by the bound workload."""

from pathlib import Path
import subprocess

import pytest
import yaml

from hyperloom.inference_optimizer.session import manifest


def checkout(path: Path, content: str) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    (path / "recipe.txt").write_text(content)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "recipe"],
        check=True,
        capture_output=True,
    )
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    return path, sha


def bind(monkeypatch, tmp_path, block):
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"benchmark": block}))
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(config))


@pytest.mark.parametrize("agentx", [True, False])
def test_manifest_records_point_checkout_instead_of_ambient_checkout(monkeypatch, tmp_path, agentx):
    ambient, ambient_sha = checkout(tmp_path / "ambient", "synthetic")
    selected, selected_sha = checkout(tmp_path / "selected", "native")
    assert ambient_sha != selected_sha
    monkeypatch.setenv("INFERENCEX_PATH", str(ambient))
    bind(monkeypatch, tmp_path, {"agentx": agentx, "inferencex_path": str(selected)})
    monkeypatch.setattr(manifest, "build_provenance", lambda *a, **k: {"code_revision": "test", "image": "test"})

    dependency = manifest.build_manifest(tmp_path / "session")["dependencies"]["inferencex"]

    assert dependency["path"] == str(selected)
    assert dependency["commit"] == selected_sha


@pytest.mark.parametrize("block", [None, {"agentx": True}, {"agentx": False, "inferencex_path": ""}])
def test_manifest_keeps_ambient_fallback_when_point_has_no_checkout(monkeypatch, tmp_path, block):
    ambient, sha = checkout(tmp_path / "ambient", "synthetic")
    monkeypatch.setenv("INFERENCEX_PATH", str(ambient))
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    if block is not None:
        bind(monkeypatch, tmp_path, block)
    assert manifest._build_dependencies()["inferencex"]["commit"] == sha


def test_missing_selected_checkout_is_recorded_without_ambient_fallback(monkeypatch, tmp_path):
    ambient, _ = checkout(tmp_path / "ambient", "synthetic")
    selected = tmp_path / "not-materialized"
    monkeypatch.setenv("INFERENCEX_PATH", str(ambient))
    bind(monkeypatch, tmp_path, {"inferencex_path": str(selected)})
    assert manifest._build_dependencies()["inferencex"] == {"path": str(selected), "commit": "", "remote": ""}


def test_no_checkout_records_unknown_dependency(monkeypatch):
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    assert manifest._build_dependencies()["inferencex"] == {"path": "", "commit": "", "remote": ""}


def test_malformed_point_is_not_replaced_by_ambient_provenance(monkeypatch, tmp_path):
    config = tmp_path / "broken.yaml"
    config.write_text("benchmark: not-a-mapping")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(config))
    with pytest.raises(ValueError, match="benchmark"):
        manifest._build_dependencies()
