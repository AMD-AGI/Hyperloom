# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile test's server runs TraceLens's patch from the test's own farm or this host's copy, never from the installed
framework."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import farm as farm_module
from hyperloom.measure.farm import FarmError, build_farm, lay_profile_patch, place_host_copies
from hyperloom.measure.spec import EnvEntry, EnvRoot
from hyperloom.orchestrator.actions.executors import PatchedServer

_RUNNER = "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
_SOURCE = "def replay(batch):\n    return batch\n"
_PATCH = """\
diff --git a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
--- a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
+++ b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,2 +1,6 @@
+def _set_profile_trace_tag(self, tag):
+    self.tag = tag
+    return "_profile_runner_name"
+
 def replay(batch):
     return batch
"""
_CANDIDATE = b"OTHER = 'candidate'\n"
_PROFILED = {"framework": "sglang", "envs": {"HYPERLOOM_TRACELENS_PATCH_STATUS": "ok"}}


@pytest.fixture
def installed(tmp_path: Path, monkeypatch) -> Path:
    """An SGLang 0.5.21 checkout installed on this host, with TraceLens's graph-capture patch for it; its site."""
    site = tmp_path / "sgl-workspace" / "sglang" / "python"
    for rel, text in {
        "sglang/__init__.py": "__version__ = '0.5.21'\n",
        "sglang/srt/__init__.py": "",
        "sglang/srt/other.py": "OTHER = 'image'\n",
        "sglang/srt/model_executor/__init__.py": "",
        "sglang/srt/model_executor/runner/__init__.py": "",
        _RUNNER: _SOURCE,
    }.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text, encoding="utf-8")
    patches = tmp_path / "TraceLens" / "examples" / "custom_workflows" / "inference_analysis" / "sglang_gc_patch"
    (patches / "sglang_0_5_21").mkdir(parents=True)
    (patches / "sglang_0_5_21" / "decode.patch").write_text(_PATCH, encoding="utf-8")
    package = types.ModuleType("sglang")
    package.__version__ = "0.5.21"  # type: ignore[attr-defined]
    package.__file__ = str(site / "sglang" / "__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", package)
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))
    monkeypatch.delenv("HYPERLOOM_SGLANG_VERSION_PIN", raising=False)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    return site


def _tree(site: Path) -> dict[str, bytes]:
    return {str(path.relative_to(site)): path.read_bytes() for path in sorted(site.rglob("*")) if path.is_file()}


def _farm(tmp_path: Path, installed: Path, monkeypatch) -> farm_module.Farm:
    """The farm of a candidate that changed one SGLang file, built as a test builds it."""
    monkeypatch.setattr(farm_module, "locate_package", lambda name, python, cwd: installed / name)
    address = "sha256:" + hashlib.sha256(_CANDIDATE).hexdigest()
    root = EnvRoot(
        "sglang", "package", "sglang", "sglang", "0.5.21", (EnvEntry("srt/other.py", "upsert", address, "0644"),)
    )
    built = build_farm([root], tmp_path / "farm", python=sys.executable, read_blob=lambda address: _CANDIDATE)
    assert built is not None
    return built


def _imported(site: Path, installed: Path) -> tuple[str, bool]:
    """Where a server with ``site`` first on its path imports the runner from, and whether that carries the patch."""
    probe = "import sglang.srt.model_executor.runner.decode_cuda_graph_runner as m; print(m.__file__, hasattr(m, '_set_profile_trace_tag'))"
    done = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join([str(site), str(installed)])},
        cwd=site.parent,
    )
    origin, patched = done.stdout.split()
    return origin, patched == "True"


def test_a_farm_that_rebuilt_the_profiled_framework_gets_the_patch_in_its_own_files(tmp_path, installed, monkeypatch):
    farm = _farm(tmp_path, installed, monkeypatch)
    before = _tree(installed)

    lay_profile_patch(farm, {"benchmark": _PROFILED})

    assert _imported(farm.site, installed) == (str(farm.site / _RUNNER), True)
    assert (farm.site / "sglang" / "srt" / "other.py").read_bytes() == _CANDIDATE, "the candidate's change is kept"
    assert _tree(installed) == before, "the installed tree is byte-identical"
    assert not (tmp_path / "tmp" / "hyperloom-tracelens").exists(), "the farm's patch is laid in the farm"


def test_a_test_that_does_not_profile_with_the_patch_leaves_its_farm_as_built(tmp_path, installed, monkeypatch):
    farm = _farm(tmp_path, installed, monkeypatch)
    monkeypatch.setattr(farm_module, "lay_tracelens_patch", lambda package, site: pytest.fail("the patch was laid"))

    lay_profile_patch(
        farm, {"benchmark": {"framework": "sglang", "envs": {"HYPERLOOM_TRACELENS_PATCH_STATUS": "unavailable"}}}
    )
    lay_profile_patch(None, {"benchmark": _PROFILED})

    assert _imported(farm.site, installed) == (str(farm.site / _RUNNER), False)


def test_a_farm_the_patch_cannot_be_laid_in_refuses_a_profile_asking_for_the_patched_fields(tmp_path, monkeypatch):
    farm = farm_module.Farm(tmp_path / "farm" / "site", ("vllm",))
    monkeypatch.setattr(farm_module, "lay_tracelens_patch", lambda package, site: False)
    asked = {"framework": "vllm", "envs": {"EXTRA_VLLM_ARGS": "--profiler-config.capture_torch_profiler True"}}

    with pytest.raises(FarmError, match="only TraceLens's patch adds"):
        lay_profile_patch(farm, {"benchmark": asked})


def _placed(tmp_path: Path, config: dict) -> tuple[dict, bool]:
    """``config`` as a profile point, after this host's copy is placed in it, and whether it was rewritten."""
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "work").mkdir(exist_ok=True)
    (placed,) = place_host_copies([point], tmp_path / "work")
    return yaml.safe_load(placed.read_text(encoding="utf-8")), placed != point


def test_this_hosts_copy_takes_the_place_of_the_copy_the_point_was_rendered_with(tmp_path, monkeypatch):
    site = tmp_path / "tmp" / "hyperloom-tracelens" / "sglang-gc-0123" / "site"
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package: PatchedServer(site))
    rendered = [
        "/farm/site",
        "/app/tmp/hyperloom-tracelens/sglang-gc-ab/site",
        "/tool",
        "/x/hyperloom-tracelens/v/site",
    ]
    config = {"benchmark": {**_PROFILED, "envs": {**_PROFILED["envs"], "PYTHONPATH": os.pathsep.join(rendered)}}}

    placed, rewritten = _placed(tmp_path, config)

    assert rewritten
    assert placed["benchmark"]["envs"]["PYTHONPATH"] == os.pathsep.join(["/farm/site", str(site), "/tool"])
    assert _placed(tmp_path, placed) == (placed, False), "a point that already names this host's copy is kept"


def test_this_hosts_copy_goes_into_the_agentx_launch_after_what_it_inherits(tmp_path, monkeypatch):
    site = tmp_path / "tmp" / "hyperloom-tracelens" / "sglang-gc-0123" / "site"
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package: PatchedServer(site))
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    config = {"benchmark": {**_PROFILED, "agentx": {"enabled": True, "launch_overrides": {"env": {}}}}}

    placed, rewritten = _placed(tmp_path, config)

    assert rewritten
    assert placed["benchmark"]["agentx"]["launch_overrides"]["env"]["PYTHONPATH"] == os.pathsep.join(
        ["/inherited", str(site)]
    )
    assert "PYTHONPATH" not in placed["benchmark"]["envs"]


def test_a_point_rendered_without_the_patch_is_kept_and_lays_no_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package: pytest.fail("a copy was laid"))
    config = {"benchmark": {"framework": "sglang", "envs": {"HYPERLOOM_TRACELENS_PATCH_STATUS": "unavailable"}}}

    assert _placed(tmp_path, config) == (config, False)


def test_a_point_asking_for_the_patched_fields_is_refused_where_no_copy_can_be_laid(tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package: None)
    asked = {"framework": "vllm", "envs": {"EXTRA_VLLM_ARGS": "--profiler-config.detailed_trace_annotation True"}}

    with pytest.raises(FarmError, match="only TraceLens's vLLM patch adds"):
        _placed(tmp_path, {"benchmark": asked})
