# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pod-side patcher applies the controller's ``--patch-set``, else gates on the pod's own SGLang version."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_RUNNER_REL = Path("python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py")
_SOURCE = """\
def capture():
    return 1


def done():
    return 0
"""
_PATCH = """\
diff --git a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
--- a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
+++ b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,2 +1,5 @@
+def _set_profile_trace_tag(self):
+    return "_profile_runner_name"
+
 def capture():
     return 1
"""


@pytest.fixture
def pod_patcher():
    import hyperloom.inference_optimizer as io_pkg

    script = Path(io_pkg.__file__).parent / "multi_node" / "scripts" / "apply_tracelens_patch_multinode.py"
    spec = importlib.util.spec_from_file_location("_tracelens_pod_patcher_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install(tmp_path: Path, monkeypatch, version: str) -> tuple[Path, Path]:
    repo = tmp_path / "sglang"
    target = repo / _RUNNER_REL
    target.parent.mkdir(parents=True)
    target.write_text(_SOURCE, encoding="utf-8")
    pkg = repo / "python" / "sglang"
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    mod = types.ModuleType("sglang")
    mod.__version__ = version
    mod.__file__ = str(pkg / "__init__.py")
    ver = types.ModuleType("sglang.version")
    ver.__version__ = version
    monkeypatch.setitem(sys.modules, "sglang", mod)
    monkeypatch.setitem(sys.modules, "sglang.version", ver)
    monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)

    tracelens = tmp_path / "TraceLens"
    gc_dir = tracelens / "examples" / "custom_workflows" / "inference_analysis" / "sglang_gc_patch" / "sglang_0_5_21"
    gc_dir.mkdir(parents=True)
    (gc_dir / "decode_cuda_graph_runner.patch").write_text(_PATCH, encoding="utf-8")
    return target, tracelens


@pytest.mark.parametrize(
    ("version", "override", "expected"),
    [
        ("0.5.17", "", False),
        ("0.5.18", "", True),
        ("0.5.21.dev3+gabc1234", "", True),
        ("0.5.21", "patch", False),
        ("0.5.10", "sitecustomize", True),
        ("", "", False),
    ],
)
def test_gate_follows_version_then_override(pod_patcher, monkeypatch, version, override, expected):
    if override:
        monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", override)
    else:
        monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)
    assert pod_patcher._uses_gc_patch(version) is expected


def test_pod_applies_gc_patch_then_skips(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.21")

    first = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert first["status"] == "applied", first
    assert first["patch_set"] == "graph-capture"
    assert "_set_profile_trace_tag" in target.read_text(encoding="utf-8")

    second = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert second["status"] == "skipped", second


def test_pod_below_gate_stays_on_roofline(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.17")

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert result["patch_set"] == "roofline"
    assert target.read_text(encoding="utf-8") == _SOURCE


def test_pod_without_exact_gc_set_skips_without_failing(pod_patcher, tmp_path, monkeypatch, capsys):
    """No gc set for the pod's version: leave the tree alone, don't borrow an older set, and don't fail the fan-out."""
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.22")

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None, patch_set="graph-capture"
    )
    assert result["status"] == "skipped"
    assert result["error"] is None
    assert "no sglang_gc_patch set" in result["skip_reason"]
    assert target.read_text(encoding="utf-8") == _SOURCE

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_tracelens_patch_multinode.py",
            "--local",
            "--tracelens-root",
            str(tracelens),
            "--patch-set",
            "graph-capture",
        ],
    )
    assert pod_patcher.main() == 0
    assert '"status": "skipped"' in capsys.readouterr().out


def test_controller_roofline_choice_wins_over_pod_version(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.21")

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None, patch_set="roofline"
    )
    assert result["patch_set"] == "roofline"
    assert target.read_text(encoding="utf-8") == _SOURCE


def test_controller_graph_capture_choice_wins_over_pod_env(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.21")
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "patch")

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None, patch_set="graph-capture"
    )
    assert result["status"] == "applied", result
    assert result["patch_set"] == "graph-capture"
    assert "_set_profile_trace_tag" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(("mode", "expected"), [("sitecustomize", "graph-capture"), ("patched", "roofline")])
def test_controller_patch_set_follows_shape_mode(monkeypatch, mode, expected):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    monkeypatch.setattr(_server_patcher, "resolve_sglang_shape_mode", lambda: mode)
    assert _server_patcher.resolve_sglang_patch_set() == expected


def test_ray_entrypoint_forwards_patch_set(monkeypatch):
    from hyperloom.inference_optimizer.multi_node import cli

    monkeypatch.setattr(cli, "_read_pod_script", lambda name: "")
    with_set = cli._build_multinode_apply_tracelens_patch_entrypoint("/tl", "", "roofline")
    assert with_set.rstrip().endswith("--tracelens-root /tl --patch-set roofline")
    assert "--patch-set" not in cli._build_multinode_apply_tracelens_patch_entrypoint("/tl", "", "")


def test_cmd_apply_tracelens_patch_submits_patch_set(monkeypatch):
    from hyperloom.inference_optimizer.multi_node import cli

    monkeypatch.setattr(cli, "_load_state", lambda: {"head_pod_ip": "10.0.0.1"})
    monkeypatch.setattr(cli, "_read_pod_script", lambda name: "")
    monkeypatch.setattr(cli, "_poll_timeout_from_args", lambda args: 1)
    submitted: list[str] = []

    def _submit(state, entrypoint, **kw):
        submitted.append(entrypoint)
        return 0, {"status": "applied", "per_pod": []}, ""

    monkeypatch.setattr(cli, "_submit_and_collect_pod_json", _submit)
    ns = argparse.Namespace(
        tracelens_root="/tl",
        sglang_version_pin=None,
        patch_set="graph-capture",
        poll_interval=1,
        print_logs=False,
    )
    assert cli.cmd_apply_tracelens_patch(ns) == cli.EXIT_OK
    assert "--patch-set graph-capture" in submitted[0]


def test_infera_forwards_patch_set(monkeypatch):
    import hyperloom.inference_optimizer.multi_node.commands.infera as inf

    monkeypatch.setattr(inf, "_infera_require_state", lambda: {"backend": "infera"})
    monkeypatch.setattr(inf, "_infera_all_gpu_targets", lambda state: [{"podIP": "10.0.1.0", "sshPort": 22}])
    monkeypatch.setattr(inf._mn_cli, "_read_pod_script", lambda name: "")
    monkeypatch.setattr(inf._mn_cli, "_poll_timeout_from_args", lambda args: 1)
    op_args: list[str] = []

    def _ssh(st, ip, script, python, args, **kw):
        op_args.append(args)
        stdout = '{"status":"applied","per_pod":[{"status":"applied"}]}'
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(inf._mn_cli, "_infera_ssh_run_script", _ssh)
    ns = argparse.Namespace(tracelens_root="/tl", sglang_version_pin="", patch_set="roofline", poll_timeout=1)
    assert inf._infera_apply_tracelens_patch(ns) == 0
    assert "--patch-set roofline" in op_args[0]


class _StopAfterFanOut(Exception):
    pass


def _fan_out_calls(monkeypatch, tmp_path: Path, *, patch_set: str, torch_profiler_dir: str) -> list[str]:
    """Run ``restart_server_for_round`` up to the fan-out and return the patch sets it fanned out."""
    from hyperloom.inference_optimizer.multi_node import cli
    from hyperloom.inference_optimizer.multi_node._internal import external_state
    from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as lifecycle
    from hyperloom.orchestrator.actions.executors import _server_patcher

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    monkeypatch.setattr(lifecycle, "is_multi_node", lambda: True)
    monkeypatch.setattr(external_state, "external_service_url", lambda: "")
    monkeypatch.setattr(_server_patcher, "resolve_sglang_patch_set", lambda: patch_set)
    calls: list[str] = []
    monkeypatch.setattr(cli, "cmd_apply_tracelens_patch", lambda ns: calls.append(ns.patch_set) or 0)

    def _stop() -> int:
        raise _StopAfterFanOut

    monkeypatch.setattr(cli, "_resolve_poll_timeout_s", _stop)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(
            lifecycle.restart_server_for_round(
                framework="sglang", model_path="/m", tp=8, ep=1, torch_profiler_dir=torch_profiler_dir
            )
        )
    chain, exc = [], excinfo.value
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    assert any(isinstance(e, _StopAfterFanOut) for e in chain), excinfo.value
    return calls


def test_gc_fan_out_waits_for_a_profiling_round(monkeypatch, tmp_path):
    assert _fan_out_calls(monkeypatch, tmp_path, patch_set="graph-capture", torch_profiler_dir="") == []
    trace_dir = str(tmp_path / "trace")
    assert _fan_out_calls(monkeypatch, tmp_path, patch_set="graph-capture", torch_profiler_dir=trace_dir) == [
        "graph-capture"
    ]


def test_roofline_fan_out_still_runs_on_every_restart(monkeypatch, tmp_path):
    assert _fan_out_calls(monkeypatch, tmp_path, patch_set="roofline", torch_profiler_dir="") == ["roofline"]
