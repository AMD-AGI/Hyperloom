# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile lays TraceLens's patch in a copy of the installed framework, which its server imports; the installed tree
stays byte-identical whatever the profile does, and profiles running at once share the copy without undoing it."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.orchestrator.actions.executors import _patch_laying, _server_patcher
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs

_PATCH_TREE = ("examples", "custom_workflows", "inference_analysis")

_SGLANG_MODULE = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
_SGLANG_RUNNER = "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
_SGLANG_SOURCE = "def run_once(graph):\n    return graph\n\n\ndef reset():\n    return None\n"
_SGLANG_PATCH = """\
diff --git a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
--- a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
+++ b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,2 +1,5 @@
+def _set_profile_trace_tag(self):
+    return "_profile_runner_name:" + str(self)
+
 def run_once(graph):
     return graph
"""

_VLLM_MODULE = "vllm.v1.worker.gpu_worker"
_VLLM_WORKER = "vllm/v1/worker/gpu_worker.py"
_VLLM_SOURCE = "def work():\n    return 1\n"
_VLLM_PATCH = """\
diff --git a/vllm/profiler/graph_capture.py b/vllm/profiler/graph_capture.py
new file mode 100644
--- /dev/null
+++ b/vllm/profiler/graph_capture.py
@@ -0,0 +1,2 @@
+def graph_capture_profiler():
+    return "capture_traces"
diff --git a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -1,2 +1,5 @@
+from vllm.profiler.graph_capture import graph_capture_profiler as _set_profile_trace_tag
+
+
 def work():
     return 1
"""


class _Install:
    """A framework installed on this host, its TraceLens patch, and the module its patch changes."""

    def __init__(self, tmp_path: Path, monkeypatch, framework: str) -> None:
        self.framework = framework
        self.tracelens = tmp_path / "TraceLens"
        if framework == "sglang":
            self.site = tmp_path / "sgl-workspace" / "sglang" / "python"
            self.files = {
                "sglang/__init__.py": "__version__ = '0.5.21'\n",
                "sglang/srt/__init__.py": "",
                "sglang/srt/other.py": "OTHER = 1\n",
                "sglang/srt/model_executor/__init__.py": "",
                "sglang/srt/model_executor/runner/__init__.py": "",
                _SGLANG_RUNNER: _SGLANG_SOURCE,
                "sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py": "PREFILL = 1\n",
            }
            self.module, self.version = _SGLANG_MODULE, "0.5.21"
            self.patch = self.tracelens.joinpath(*_PATCH_TREE, "sglang_gc_patch", "sglang_0_5_21", "decode.patch")
            self.patch_text = _SGLANG_PATCH
        else:
            self.site = tmp_path / "venv" / "lib" / "python3.10" / "site-packages"
            self.files = {
                "vllm/__init__.py": "__version__ = '0.26.0'\n",
                "vllm/config/__init__.py": "",
                "vllm/config/profiler.py": "capture_torch_profiler = False\ndetailed_trace_annotation = False\n",
                "vllm/profiler/__init__.py": "",
                "vllm/v1/__init__.py": "",
                "vllm/v1/worker/__init__.py": "",
                _VLLM_WORKER: _VLLM_SOURCE,
            }
            self.module, self.version = _VLLM_MODULE, "0.26.0"
            self.patch = self.tracelens.joinpath(*_PATCH_TREE, "vllm_patches", "config_vllm_v0.26.0.patch")
            self.patch_text = _VLLM_PATCH
        for rel, text in self.files.items():
            (self.site / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.site / rel).write_text(text, encoding="utf-8")
        self.patch.parent.mkdir(parents=True)
        self.patch.write_text(self.patch_text, encoding="utf-8")
        package = types.ModuleType(framework)
        package.__version__ = self.version  # type: ignore[attr-defined]
        package.__file__ = str(self.site / framework / "__init__.py")
        monkeypatch.setitem(sys.modules, framework, package)
        monkeypatch.setenv("TRACELENS_ROOT", str(self.tracelens))

    def snapshot(self) -> dict[str, tuple[str, bytes, int]]:
        """Every entry below the installed site: its kind, its bytes and its mode."""
        found = {}
        for path in sorted(self.site.rglob("*")):
            kind = "link" if path.is_symlink() else "dir" if path.is_dir() else "file"
            data = path.read_bytes() if kind == "file" else b""
            found[str(path.relative_to(self.site))] = (kind, data, stat.S_IMODE(path.lstat().st_mode))
        return found


@pytest.fixture(autouse=True)
def profile_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    monkeypatch.delenv("HYPERLOOM_SGLANG_VERSION_PIN", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    for name in (
        "PROFILE_OSL",
        "HYPERLOOM_PROFILE_MAX_ITERS",
        "HYPERLOOM_PROFILE_DELAY_ITERS",
        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY",
        "HYPERLOOM_VLLM_IMAGE_SOURCE",
        "ISL",
        "OSL",
        "CONC",
        "TP",
    ):
        monkeypatch.delenv(name, raising=False)


def _copies(tmp_path: Path) -> list[Path]:
    root = tmp_path / "tmp" / "hyperloom-tracelens"
    return sorted(root.iterdir()) if root.is_dir() else []


def _profile(tmp_path: Path, framework: str) -> dict:
    """The server variables of the shipped profile configuration of ``framework``, materialized."""
    source = asset_root() / "assets" / "configs" / f"profile_{framework}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["benchmark"]["model"] = "Qwen/Qwen3-0.6B"
    config["benchmark"]["envs"].update({"TP": 1, "CONC": 4, "ISL": 256, "OSL": 8192})
    path = tmp_path / f"profile_{framework}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rendered = materialize_config_with_envs(path, tmp_path / f"run_{framework}")
    return yaml.safe_load(rendered.read_text(encoding="utf-8"))["benchmark"]["envs"]


def _serve(install: _Install, pythonpath: str) -> tuple[str, bool]:
    """A stub server started with the profile's ``PYTHONPATH``, the installed site after it as Python would add it:
    the file its patched module came from, and whether it carries the patch."""
    probe = (
        "import importlib, sys\n"
        "module = importlib.import_module(sys.argv[1])\n"
        "print(module.__file__)\n"
        "print(hasattr(module, '_set_profile_trace_tag'))\n"
    )
    done = subprocess.run(
        [sys.executable, "-S", "-c", probe, install.module],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join([pythonpath, str(install.site)])},
        cwd=install.site.parent,
    )
    origin, patched = done.stdout.split()
    return origin, patched == "True"


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_a_profiles_server_imports_the_patched_copy_and_the_installed_tree_is_left_as_it_was(
    tmp_path, monkeypatch, framework
):
    install = _Install(tmp_path, monkeypatch, framework)
    before = install.snapshot()

    envs = _profile(tmp_path, framework)

    assert envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "ok"
    (copy,) = _copies(tmp_path)
    site = copy / "site"
    assert envs["PYTHONPATH"].split(os.pathsep)[0] == str(site)
    origin, patched = _serve(install, envs["PYTHONPATH"])
    assert patched and Path(origin).is_relative_to(site), "the server runs the patched copy"
    assert install.snapshot() == before, "the installed tree is byte-identical"
    assert sorted(path.name for path in site.iterdir()) == [framework], "the copy shadows nothing else"


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_a_profile_whose_patch_does_not_apply_leaves_the_installed_tree_and_keeps_no_copy(
    tmp_path, monkeypatch, framework
):
    install = _Install(tmp_path, monkeypatch, framework)
    drifted = install.site / (_SGLANG_RUNNER if framework == "sglang" else _VLLM_WORKER)
    drifted.write_text(drifted.read_text(encoding="utf-8").replace("    return", "    return 2 *"), encoding="utf-8")
    before = install.snapshot()

    envs = _profile(tmp_path, framework)

    assert envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "unavailable"
    assert not any(_patch_laying.is_patched_copy(part) for part in str(envs.get("PYTHONPATH", "")).split(os.pathsep))
    assert install.snapshot() == before
    assert _copies(tmp_path) == [], "nothing half-built is left"


class _Stopped(Exception):
    """The profile is stopped half way through laying the patch."""


def test_a_profile_that_raises_while_patching_leaves_the_installed_tree_and_keeps_no_copy(tmp_path, monkeypatch):
    install = _Install(tmp_path, monkeypatch, "sglang")
    before = install.snapshot()
    real_git = _patch_laying._git

    def _git(git: str, args, cwd: Path) -> bool:
        applied = real_git(git, args, cwd)
        if "--check" not in args:
            raise _Stopped
        return applied

    monkeypatch.setattr(_patch_laying, "_git", _git)

    with pytest.raises(_Stopped):
        _profile(tmp_path, "sglang")

    assert install.snapshot() == before
    assert _copies(tmp_path) == []


def test_two_profiles_at_once_share_one_copy_and_neither_undoes_the_other(tmp_path, monkeypatch):
    install = _Install(tmp_path, monkeypatch, "sglang")
    before = install.snapshot()
    start = threading.Barrier(2)
    sites: list[Path | None] = []

    def profile() -> None:
        start.wait()
        patched = _server_patcher.ensure_sglang_patched_for_tracelens()
        sites.append(None if patched is None else patched.site)

    threads = [threading.Thread(target=profile) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (site,) = set(sites)
    assert site is not None and _copies(tmp_path) == [site.parent]
    laid = {path: path.read_bytes() for path in site.rglob("*.py") if not path.is_symlink()}
    assert site / _SGLANG_RUNNER in laid

    # A third profile, whose patch set changed and no longer applies, ends while the first two still run.
    install.patch.write_text(_SGLANG_PATCH.replace("     return graph", "     return None"), encoding="utf-8")
    assert _server_patcher.ensure_sglang_patched_for_tracelens() is None

    assert {path: path.read_bytes() for path in laid} == laid, "no profile writes the copy another one runs"
    assert _serve(install, str(site)) == (str(site / _SGLANG_RUNNER), True)
    assert install.snapshot() == before


def test_a_farm_that_rebuilt_the_framework_gets_the_patch_in_its_own_files_only(tmp_path, monkeypatch):
    install = _Install(tmp_path, monkeypatch, "sglang")
    before = install.snapshot()
    farm = tmp_path / "farm" / "site"
    _patch_laying._mirror(install.site / "sglang", farm / "sglang")
    (farm / "sglang" / "srt").unlink()
    _patch_laying._mirror(install.site / "sglang" / "srt", farm / "sglang" / "srt")
    candidate = farm / "sglang" / "srt" / "other.py"
    candidate.unlink()
    candidate.write_text("OTHER = 2\n", encoding="utf-8")

    assert _server_patcher.lay_tracelens_patch("sglang", farm) is True

    assert _serve(install, str(farm)) == (str(farm / _SGLANG_RUNNER), True)
    assert candidate.read_text(encoding="utf-8") == "OTHER = 2\n", "the candidate's own file is kept"
    assert (farm / "sglang" / "srt" / "model_executor").is_dir() and not (
        farm / "sglang" / "srt" / "model_executor"
    ).is_symlink()
    assert install.snapshot() == before
    assert _copies(tmp_path) == [], "a farm's patch is laid in the farm"


def test_the_copy_goes_ahead_of_a_vllm_source_checkout_the_launch_puts_on_the_path(tmp_path, monkeypatch):
    _Install(tmp_path, monkeypatch, "vllm")
    checkout = tmp_path / "vllm-checkout"
    checkout.mkdir()
    monkeypatch.setenv("HYPERLOOM_VLLM_IMAGE_SOURCE", "1")
    for name in ("VLLM_REPO_PATH", "FRAMEWORK_REPO_PATH", "VLLM_DIR"):
        monkeypatch.setenv(name, str(checkout))

    envs = _profile(tmp_path, "vllm")

    (copy,) = _copies(tmp_path)
    assert envs["PYTHONPATH"].split(os.pathsep)[:2] == [str(copy / "site"), str(checkout)]
