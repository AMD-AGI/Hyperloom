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
from hyperloom.measure.farm import (
    PATCH_NOT_LAID,
    FarmError,
    build_farm,
    lay_profile_patch,
    place_host_copies,
    record_unpatched,
)
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


def _farm(
    tmp_path: Path, installed: Path, monkeypatch, changed: str = "srt/other.py", data: bytes = _CANDIDATE
) -> farm_module.Farm:
    """The farm of a candidate that changed one SGLang file (``changed`` to ``data``), built as a test builds it."""
    monkeypatch.setattr(farm_module, "locate_package", lambda name, python, cwd: installed / name)
    address = "sha256:" + hashlib.sha256(data).hexdigest()
    root = EnvRoot("sglang", "package", "sglang", "sglang", "0.5.21", (EnvEntry(changed, "upsert", address, "0644"),))
    built = build_farm([root], tmp_path / "farm", python=sys.executable, read_blob=lambda address: data)
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

    assert lay_profile_patch(farm, {"benchmark": _PROFILED}) is None

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


def test_a_farm_the_graph_capture_patch_cannot_be_laid_in_runs_the_profile_and_its_record_says_so(
    tmp_path, installed, monkeypatch
):
    """The candidate rewrote the runner the patch writes, so the patch no longer applies there: the server runs the
    candidate's runner unpatched, and the test's note and every point say the patch is not there."""
    drifted = _SOURCE.replace("    return batch", "    return 2 * batch").encode()
    farm = _farm(tmp_path, installed, monkeypatch, changed=_RUNNER.removeprefix("sglang/"), data=drifted)
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {**_PROFILED, "envs": {**_PROFILED["envs"], "TP": 1}}}), "utf-8")

    note = lay_profile_patch(farm, {"benchmark": _PROFILED})
    record_unpatched(point)

    assert note is not None and note.startswith(f"{PATCH_NOT_LAID}: ") and "the sglang the test rebuilt" in note
    assert _imported(farm.site, installed) == (str(farm.site / _RUNNER), False)
    assert (farm.site / _RUNNER).read_bytes() == drifted, "the candidate's runner is kept as it is"
    assert yaml.safe_load(point.read_text(encoding="utf-8"))["benchmark"]["envs"] == {
        "HYPERLOOM_TRACELENS_PATCH_STATUS": "unavailable",
        "TP": 1,
    }


@pytest.mark.parametrize(
    ("framework", "variable", "flag"),
    [
        ("vllm", "EXTRA_VLLM_ARGS", "--profiler-config.capture_torch_profiler"),
        ("sglang", "EXTRA_SGLANG_ARGS", "--enable-shape-discovery-for-cuda-graph-profile"),
    ],
)
def test_a_farm_the_patch_cannot_be_laid_in_refuses_a_profile_passing_its_server_a_flag_only_the_patch_adds(
    tmp_path, monkeypatch, framework, variable, flag
):
    farm = farm_module.Farm(tmp_path / "farm" / "site", (framework,))
    monkeypatch.setattr(farm_module, "lay_tracelens_patch", lambda package, site: False)
    asked = {"framework": framework, "envs": {"HYPERLOOM_TRACELENS_PATCH_STATUS": "ok", variable: f"{flag} True"}}

    with pytest.raises(FarmError, match=f"asks its server for {flag}, which only TraceLens's patch adds"):
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
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: PatchedServer(site))
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


@pytest.mark.parametrize("on_the_path", [True, False])
def test_this_hosts_copy_of_a_vllm_point_mirrors_the_image_checkout_its_render_put_on_the_path(
    tmp_path, monkeypatch, on_the_path
):
    site = tmp_path / "tmp" / "hyperloom-tracelens" / "vllm-0123" / "site"
    laid: list[tuple[str, Path | None]] = []
    monkeypatch.setattr(
        farm_module,
        "ensure_patched_copy",
        lambda package, checkout: laid.append((package, checkout)) or PatchedServer(site),
    )
    checkout = tmp_path / "app" / "vllm"
    rendered = ["/app/tmp/hyperloom-tracelens/vllm-ab/site", *([str(checkout)] if on_the_path else []), "/tool"]
    envs = {"HYPERLOOM_TRACELENS_PATCH_STATUS": "ok", "VLLM_REPO_PATH": str(checkout)}
    config = {"benchmark": {"framework": "vllm", "envs": {**envs, "PYTHONPATH": os.pathsep.join(rendered)}}}

    placed, rewritten = _placed(tmp_path, config)

    assert rewritten and laid == [("vllm", checkout if on_the_path else None)]
    assert placed["benchmark"]["envs"]["PYTHONPATH"].split(os.pathsep) == [str(site), *rendered[1:]]


def test_this_hosts_copy_goes_into_the_agentx_launch_after_what_it_inherits(tmp_path, monkeypatch):
    site = tmp_path / "tmp" / "hyperloom-tracelens" / "sglang-gc-0123" / "site"
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: PatchedServer(site))
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    config = {"benchmark": {**_PROFILED, "agentx": {"enabled": True, "launch_overrides": {"env": {}}}}}

    placed, rewritten = _placed(tmp_path, config)

    assert rewritten
    assert placed["benchmark"]["agentx"]["launch_overrides"]["env"]["PYTHONPATH"] == os.pathsep.join(
        ["/inherited", str(site)]
    )
    assert "PYTHONPATH" not in placed["benchmark"]["envs"]


def test_a_point_rendered_without_the_patch_is_kept_and_lays_no_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: pytest.fail("a copy was laid"))
    config = {"benchmark": {"framework": "sglang", "envs": {"HYPERLOOM_TRACELENS_PATCH_STATUS": "unavailable"}}}

    assert _placed(tmp_path, config) == (config, False)


def test_a_point_asking_for_the_patched_fields_is_refused_where_no_copy_can_be_laid(tmp_path, monkeypatch):
    monkeypatch.setattr(farm_module, "ensure_patched_copy", lambda package, checkout: None)
    asked = {"framework": "vllm", "envs": {"EXTRA_VLLM_ARGS": "--profiler-config.detailed_trace_annotation True"}}

    with pytest.raises(
        FarmError, match="only TraceLens's patch adds, and the patch cannot be applied to this host's vllm"
    ):
        _placed(tmp_path, {"benchmark": asked})
