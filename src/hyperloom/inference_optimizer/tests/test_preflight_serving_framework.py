# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight serving-framework importability gate."""

from __future__ import annotations

import argparse
import re
import sys

import pytest

from hyperloom.inference_optimizer.cli import preflight

_SKIP_ENV = "HYPERLOOM_SKIP_FRAMEWORK_CHECK"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        _SKIP_ENV,
        "FRAMEWORK",
        "BENCHMARK_BASE_URL",
        "VLLM_VENV_ROOT",
        "HYPERLOOM_MN_EXT_SERVICE_URL",
        "INFERENCE_OPTIMIZER_NODES",
    ):
        monkeypatch.delenv(key, raising=False)


def _args(framework: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(framework=framework)


def _probe_result(monkeypatch, importable: bool, *, rocm: bool | None = True) -> list[list[str]]:
    """Stub the interpreter probe; return the recorded argv list."""
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0 if importable else 1
        stdout = ""
        stderr = ""

    def _run(cmd, *_args, **_kwargs):
        calls.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(preflight.subprocess, "run", _run)
    monkeypatch.setattr(preflight, "_probe_rocm_build", lambda _fw, _py: rocm)
    return calls


def _probe_per_interpreter(monkeypatch, table, default=(False, None)) -> list[list[str]]:
    """Stub importability and the ROCm verdict per interpreter path."""
    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, rc: int) -> None:
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def _run(cmd, *_args, **_kwargs):
        calls.append(list(cmd))
        return _Proc(0 if table.get(cmd[0], default)[0] else 1)

    monkeypatch.setattr(preflight.subprocess, "run", _run)
    monkeypatch.setattr(preflight, "_probe_rocm_build", lambda _fw, py: table.get(py, default)[1])
    return calls


def _refuse_probe(monkeypatch) -> None:
    """Make any subprocess call an error, so an exemption can be proven."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("framework probe must not run here")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)


def test_missing_serving_framework_exits_with_guidance(monkeypatch, capsys):
    """A host with no importable serving framework must fail at preflight.

    This is the #1141 case: the run otherwise proceeds and dies much later in
    an unrelated-looking place, hiding the fact that the framework is absent.
    """
    _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    # Names the framework and both remedies, and disambiguates the two run modes
    # so nobody repeats the issue author's reading of Magpie's run_mode=local.
    assert "vllm" in err
    assert "--install-framework vllm" in err
    assert "HYPERLOOM_RUN_MODE=docker" in err
    assert _SKIP_ENV in err


@pytest.mark.parametrize("in_container", [False, True])
def test_guidance_carries_no_image_tags_or_doc_paths(monkeypatch, capsys, in_container):
    """Error text must only name things that cannot go stale or go missing.

    An image tag encodes framework, ROCm and GPU-arch versions that all move
    independently, and no test executes this branch when they bump. A repo doc
    path is worse: ``pip install`` ships no ``docs/``, so it dangles.
    """
    _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: in_container)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    err = capsys.readouterr().err
    assert not re.search(r"v\d+\.\d+|rocm\d|mi\d00x", err), f"stale-able version in: {err}"
    assert not re.search(r"\bdocs/\S+\.md\b", err), f"repo doc path in: {err}"
    # Still actionable: the command and the env var both exist in any install.
    assert "--install-framework vllm" in err


def test_importable_framework_proceeds(monkeypatch, capsys):
    _probe_result(monkeypatch, importable=True)

    preflight._check_serving_framework(_args("sglang"), "/usr/bin/python3")

    assert "sglang" in capsys.readouterr().out


def test_scriptable_framework_is_exempt(monkeypatch):
    """xDiT/custom own their entrypoint; no serving package is required."""
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args("xdit"), "/usr/bin/python3")


def test_remote_client_is_exempt(monkeypatch):
    """With BENCHMARK_BASE_URL the server is remote, so nothing local is needed."""
    monkeypatch.setenv("BENCHMARK_BASE_URL", "http://serving-host:8888")
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")


def test_external_multi_node_is_exempt(monkeypatch):
    """Mirrors the GPU-visibility skip: serving happens on remote pods."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://claw-rayjob:8000")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")


def test_escape_hatch_is_exempt(monkeypatch, capsys):
    """Parity with install_baremetal.sh's --skip-base-check."""
    monkeypatch.setenv(_SKIP_ENV, "1")
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert _SKIP_ENV in capsys.readouterr().out


def test_isolated_vllm_venv_is_probed(monkeypatch):
    """vLLM installs into $VLLM_VENV_ROOT, invisible to the benchmark python."""
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/hyperloom/vllm-venv")
    calls = _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    probed = [c[0] for c in calls]
    assert "/opt/hyperloom/vllm-venv/bin/python" in probed


def test_container_message_omits_the_container_remedy(monkeypatch, capsys):
    """Inside a container, "go run in a container" is useless advice."""
    _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: True)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    err = capsys.readouterr().err
    assert "already runs in a container" in err
    assert "HYPERLOOM_RUN_MODE=docker" not in err


def test_framework_falls_back_to_env_then_default(monkeypatch):
    """Resolution mirrors _ensure_framework_deps: args > $FRAMEWORK > default."""
    monkeypatch.setenv("FRAMEWORK", "xdit")
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args(None), "/usr/bin/python3")


def test_cuda_build_exits_even_though_importable(monkeypatch, capsys):
    """``pip install vllm`` from PyPI yields an importable CUDA build.

    Importability alone would wave it through, and it then dies at GPU init
    with no hint that the wheel is simply the wrong one.
    """
    _probe_result(monkeypatch, importable=True, rocm=False)

    with pytest.raises(SystemExit) as excinfo:
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "NOT a ROCm build" in err
    assert "PyPI" in err
    assert _SKIP_ENV in err


def test_inconclusive_rocm_probe_warns_but_proceeds(monkeypatch, capsys):
    """A probe that cannot answer must not block an otherwise valid run."""
    _probe_result(monkeypatch, importable=True, rocm=None)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    out = capsys.readouterr().out
    assert "could not verify" in out


def test_rocm_build_proceeds_quietly(monkeypatch, capsys):
    _probe_result(monkeypatch, importable=True, rocm=True)

    preflight._check_serving_framework(_args("sglang"), "/usr/bin/python3")

    out = capsys.readouterr().out
    assert "ROCm" in out
    assert "could not verify" not in out


_ISOLATED_VENV = "/opt/hyperloom/vllm-venv"
_ISOLATED_PY = f"{_ISOLATED_VENV}/bin/python"


def test_isolated_venv_rocm_build_beats_a_stray_cuda_wheel(monkeypatch, capsys):
    """Stopping at the first importable candidate rejects a working host.

    install_baremetal.sh defaults vLLM to an isolated venv, so the ROCm build
    routinely lives there while the benchmark interpreter may still carry a
    stray PyPI CUDA wheel. Every candidate has to be considered.
    """
    monkeypatch.setenv("VLLM_VENV_ROOT", _ISOLATED_VENV)
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), _ISOLATED_PY: (True, True)},
    )

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert _ISOLATED_PY in capsys.readouterr().out


def test_isolated_venv_is_probed_before_the_benchmark_interpreter(monkeypatch, capsys):
    """Mirror install_baremetal.sh, which switches the probe to the venv."""
    monkeypatch.setenv("VLLM_VENV_ROOT", _ISOLATED_VENV)
    calls = _probe_per_interpreter(monkeypatch, {_ISOLATED_PY: (True, True)})

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert calls[0][0] == _ISOLATED_PY


def test_every_importable_candidate_cuda_still_fails(monkeypatch, capsys):
    """Scanning all candidates must not weaken the gate when all are wrong."""
    monkeypatch.setenv("VLLM_VENV_ROOT", _ISOLATED_VENV)
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), _ISOLATED_PY: (True, False)},
    )

    with pytest.raises(SystemExit) as excinfo:
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert excinfo.value.code == 2
    assert "NOT a ROCm build" in capsys.readouterr().err


def test_an_inconclusive_candidate_outweighs_a_refuted_one(monkeypatch, capsys):
    """Block only when every importable candidate is provably the wrong build."""
    monkeypatch.setenv("VLLM_VENV_ROOT", _ISOLATED_VENV)
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), _ISOLATED_PY: (True, None)},
    )

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert "could not verify" in capsys.readouterr().out


def test_probe_rocm_build_reports_tri_state(monkeypatch):
    """The probe distinguishes verified / refuted / inconclusive."""

    class _Proc:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: _Proc(0))
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3") is True

    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: _Proc(1))
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3") is False

    def _boom(*_a, **_k):
        raise OSError("probe crashed")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3") is None

    # Absent torch and a signal death are "cannot answer", not "wrong build":
    # calling either a CUDA wheel would be a wrong diagnosis.
    for rc in (3, -11):
        monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, _rc=rc, **_k: _Proc(_rc))
        assert preflight._probe_rocm_build("vllm", "/usr/bin/python3") is None


def test_probe_is_inconclusive_when_torch_is_absent(monkeypatch):
    """A host with no torch must not be told its wheel is the CUDA build."""
    verdict = preflight._probe_rocm_build("sglang", sys.executable)

    assert verdict is None or verdict is True
