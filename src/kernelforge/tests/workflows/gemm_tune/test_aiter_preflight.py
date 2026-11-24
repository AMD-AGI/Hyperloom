# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""aiter tune/serve alignment preflight."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

from kernelforge.workflows.gemm_tune.aiter_preflight import classify, collect, is_aligned, serve_aiter_path


@pytest.mark.parametrize(
    ("serve", "root", "aligned"),
    [
        ("/opt/aiter", "/opt/aiter", True),
        ("/opt/aiter/aiter", "/opt/aiter", True),
        ("/opt/aiter/aiter", "/opt/aiter/", True),
        ("/usr/local/lib/python3.12/dist-packages/aiter", "/root/aiter-src", False),
        ("/opt/aiter-other/aiter", "/opt/aiter", False),
    ],
    ids=[
        "exact",
        "editable-install",
        "editable-install-trailing-slash",
        "different-trees",
        "prefix-not-a-path-boundary",
    ],
)
def test_is_aligned_for_a_source_tree(serve, root, aligned):
    assert is_aligned(serve, root) is aligned


@pytest.mark.parametrize(
    ("serve", "root", "aligned"),
    [
        ("/usr/local/lib/python3.12/dist-packages/aiter", "/usr/local/lib/python3.12/dist-packages/aiter_meta", True),
        ("/usr/local/lib/python3.12/dist-packages/aiter", "/usr/local/lib/python3.12/dist-packages/aiter_meta/", True),
        ("/opt/venv/lib/aiter", "/usr/lib/aiter_meta", False),
        ("/opt/sp/aiter", "/opt/sp/aiter_meta_old", False),
    ],
    ids=["sibling-aiter-meta", "sibling-aiter-meta-trailing-slash", "aiter-meta-of-another-tree", "similar-name"],
)
def test_is_aligned_for_a_wheel_layout(serve, root, aligned):
    # The wheel ships `aiter` (importable) and `aiter_meta` (csrc/tuners) as siblings, and resolve_aiter_root() picks
    # aiter_meta on purpose.
    assert is_aligned(serve, root) is aligned


def test_classify_aligned_clean():
    hard, soft = classify("/opt/aiter/aiter", "/opt/aiter", "abc123")
    assert hard == []
    assert soft == []


def test_classify_misaligned_is_hard():
    hard, soft = classify("/usr/local/.../aiter", "/root/aiter-src", "abc123")
    assert any("MISALIGNED" in m for m in hard)


def test_classify_unset_root_and_commit_are_soft():
    hard, soft = classify("/opt/aiter/aiter", None, None)
    assert hard == []
    assert any("AITER_ROOT_DIR" in m for m in soft)
    assert any("AITER_COMMIT" in m for m in soft)


def test_classify_no_serving_aiter_is_hard():
    hard, _ = classify(None, "/opt/aiter", "abc123")
    assert any("not importable" in m for m in hard)


def test_collect_aligned(monkeypatch, tmp_path):
    root = tmp_path / "aiter_src"
    root.mkdir()
    serve = root / "aiter"
    serve.mkdir()
    monkeypatch.setattr(
        "kernelforge.workflows.gemm_tune.aiter_preflight.serve_aiter_path", lambda: os.path.realpath(str(serve))
    )
    st = collect({"AITER_ROOT_DIR": str(root), "AITER_COMMIT": "abc123"})
    assert st["aligned"] is True
    assert st["hard"] == [] and st["soft"] == []
    assert st["aiter_commit"] == "abc123"


def test_collect_misaligned_reports_hard(monkeypatch, tmp_path):
    root = tmp_path / "aiter_src"
    root.mkdir()
    monkeypatch.setattr("kernelforge.workflows.gemm_tune.aiter_preflight.serve_aiter_path", lambda: "/usr/local/aiter")
    st = collect({"AITER_ROOT_DIR": str(root)})
    assert st["aligned"] is False
    assert any("MISALIGNED" in h for h in st["hard"])
    assert any("AITER_COMMIT" in s for s in st["soft"])  # commit unset -> soft warn


def test_collect_falls_back_to_package_version_when_commit_unset(monkeypatch, tmp_path):
    # AITER_COMMIT unset must still yield a real provenance pin (not None), and the value must be unmistakably a
    # distribution version, never a fake sha.
    from kernelforge.workflows.gemm_tune import aiter_preflight as ap

    root = tmp_path / "aiter_src"
    root.mkdir()
    (root / "aiter").mkdir()
    monkeypatch.setattr(ap, "serve_aiter_path", lambda: os.path.realpath(str(root / "aiter")))
    monkeypatch.setattr(ap, "_installed_aiter_version", lambda: "amd-aiter==0.1.13.post1")
    st = collect({"AITER_ROOT_DIR": str(root)})
    assert st["aiter_commit"] == "amd-aiter==0.1.13.post1"
    # the operator is still nudged to set an exact commit
    assert any("AITER_COMMIT" in s for s in st["soft"])


def test_collect_commit_env_wins_over_package_version(monkeypatch, tmp_path):
    from kernelforge.workflows.gemm_tune import aiter_preflight as ap

    root = tmp_path / "aiter_src"
    root.mkdir()
    (root / "aiter").mkdir()
    monkeypatch.setattr(ap, "serve_aiter_path", lambda: os.path.realpath(str(root / "aiter")))
    monkeypatch.setattr(ap, "_installed_aiter_version", lambda: "amd-aiter==0.1.13.post1")
    st = collect({"AITER_ROOT_DIR": str(root), "AITER_COMMIT": "abc123"})
    assert st["aiter_commit"] == "abc123"
    assert st["soft"] == []


def test_collect_commit_is_none_when_nothing_resolvable(monkeypatch, tmp_path):
    from kernelforge.workflows.gemm_tune import aiter_preflight as ap

    root = tmp_path / "aiter_src"
    root.mkdir()
    (root / "aiter").mkdir()
    monkeypatch.setattr(ap, "serve_aiter_path", lambda: os.path.realpath(str(root / "aiter")))
    monkeypatch.setattr(ap, "_installed_aiter_version", lambda: None)
    assert collect({"AITER_ROOT_DIR": str(root)})["aiter_commit"] is None


def test_collect_wheel_layout_is_not_reported_as_misaligned(monkeypatch, tmp_path):
    # End-to-end guard for the false HARD alarm: a wheel-shaped install must come back aligned with no hard problems.
    from kernelforge.workflows.gemm_tune import aiter_preflight as ap

    sp = tmp_path / "dist-packages"
    (sp / "aiter").mkdir(parents=True)
    (sp / "aiter_meta" / "csrc").mkdir(parents=True)
    monkeypatch.setattr(ap, "serve_aiter_path", lambda: os.path.realpath(str(sp / "aiter")))
    st = collect({"AITER_ROOT_DIR": str(sp / "aiter_meta"), "AITER_COMMIT": "abc123"})
    assert st["aligned"] is True
    assert st["hard"] == []


def test_collect_no_serve_aiter_is_hard(monkeypatch):
    monkeypatch.setattr("kernelforge.workflows.gemm_tune.aiter_preflight.serve_aiter_path", lambda: None)
    st = collect({})
    assert st["aligned"] is False
    assert any("not importable" in h for h in st["hard"])


def _loaded_without_a_spec(name):
    raise ValueError(f"{name}.__spec__ is None")


@pytest.mark.parametrize(
    "find_spec",
    [lambda name: None, lambda name: SimpleNamespace(origin=None), _loaded_without_a_spec],
    ids=["absent", "namespace-package", "loaded-module-without-a-spec"],
)
def test_serve_aiter_path_none_without_a_single_location(monkeypatch, find_spec):
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert serve_aiter_path() is None


def test_serve_aiter_path_propagates_unexpected_find_spec_errors(monkeypatch):
    def boom(name):
        raise RuntimeError("meta path finder exploded")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    with pytest.raises(RuntimeError, match="meta path finder exploded"):
        serve_aiter_path()


def test_serve_aiter_path_returns_package_dir(monkeypatch, tmp_path):
    init = tmp_path / "aiter" / "__init__.py"
    init.parent.mkdir()
    init.write_text("")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(init)))
    assert serve_aiter_path() == os.path.realpath(str(init.parent))


def test_splitk_trial_key_matches_blockscale_tuner():
    # #2 guard: the per-shape-trial gate constant must stay in sync with the tuner that actually passes --splitK, or
    # the trial silently falls back to static cap.
    from kernelforge.workflows.gemm_tune.tuners._aiter_dense_common import SPLITK_TRIAL_SCRIPT_KEY
    from kernelforge.workflows.gemm_tune.tuners.a8w8_blockscale import A8W8BlockscaleTuner

    assert A8W8BlockscaleTuner.name == SPLITK_TRIAL_SCRIPT_KEY
