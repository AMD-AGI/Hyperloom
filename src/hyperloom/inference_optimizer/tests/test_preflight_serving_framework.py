# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight serving-framework importability gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import pytest

from hyperloom.common import provenance
from hyperloom.common.provenance import (
    RESOLVED_FRAMEWORK_ENV,
    RESOLVED_FRAMEWORK_PYTHON_ENV,
)
from hyperloom.inference_optimizer.cli import preflight

_SKIP_ENV = "HYPERLOOM_SKIP_FRAMEWORK_CHECK"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        _SKIP_ENV,
        "FRAMEWORK",
        "BENCHMARK_BASE_URL",
        "VLLM_VENV_ROOT",
        "FRAMEWORK_ENV",
        "HYPERLOOM_MN_EXT_SERVICE_URL",
        "INFERENCE_OPTIMIZER_NODES",
        "KUBERNETES_SERVICE_HOST",
        "HYPERLOOM_IMAGE",
        # The check publishes these; without the reset they leak into every
        # later test in the session and into provenance lookups.
        RESOLVED_FRAMEWORK_PYTHON_ENV,
        RESOLVED_FRAMEWORK_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    # ``delenv`` records no undo for a key that was absent, and the check writes
    # these through ``os.environ`` rather than the fixture, so monkeypatch never
    # sees them. Without this they outlive the file.
    for key in (RESOLVED_FRAMEWORK_PYTHON_ENV, RESOLVED_FRAMEWORK_ENV):
        os.environ.pop(key, None)


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
    monkeypatch.setattr(preflight, "_probe_rocm_build", lambda _fw, _py: preflight._Probe(rocm))
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
    monkeypatch.setattr(preflight, "_probe_rocm_build", lambda _fw, py: preflight._Probe(table.get(py, default)[1]))
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


def test_isolated_vllm_venv_is_probed(isolated_vllm, monkeypatch):
    """vLLM installs into $VLLM_VENV_ROOT, invisible to the benchmark python."""
    calls = _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert isolated_vllm in [c[0] for c in calls]


def test_a_venv_root_without_a_python_is_not_probed(monkeypatch):
    """The variable can outlive the venv, and a stale path is not a candidate."""
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/hyperloom/vllm-venv-that-is-gone")

    probed = preflight._framework_probe_interpreters("vllm", "/usr/bin/python3")

    assert not any("vllm-venv" in candidate for candidate in probed)


def test_the_install_mode_flag_does_not_gate_the_probe(isolated_vllm, monkeypatch):
    """The installer keeps that flag under a name nothing else reads.

    install_baremetal.sh persists it as HYPERLOOM_FRAMEWORK_ENV, and only
    env_safety mentions it; framework.paths discovers the venv without it. Gating
    on a plain FRAMEWORK_ENV made this probe dead code and sent a host that had
    just installed vLLM back to the command it had already run.
    """
    monkeypatch.delenv("FRAMEWORK_ENV", raising=False)
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK_ENV", raising=False)

    probed = preflight._framework_probe_interpreters("vllm", "/usr/bin/python3")

    assert probed[0] == isolated_vllm


def test_no_other_framework_probes_the_vllm_venv(isolated_vllm):
    """That venv holds vLLM only, so sglang has no business being looked for there."""
    probed = preflight._framework_probe_interpreters("sglang", "/usr/bin/python3")

    assert isolated_vllm not in probed


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


@pytest.fixture
def isolated_vllm(monkeypatch, tmp_path):
    """The shape install_baremetal.sh switches its own probe on, and returns its python.

    $VLLM_VENV_ROOT alone is not it: the installer also requires
    FRAMEWORK_ENV=isolated and an executable python in that venv. The executable
    is real rather than a stubbed os.access, which would answer for every other
    caller in this module too.
    """
    venv = tmp_path / "vllm-venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv))

    return str(python)


def test_isolated_venv_rocm_build_beats_a_stray_cuda_wheel(isolated_vllm, monkeypatch, capsys):
    """Stopping at the first importable candidate rejects a working host.

    install_baremetal.sh defaults vLLM to an isolated venv, so the ROCm build
    routinely lives there while the benchmark interpreter may still carry a
    stray PyPI CUDA wheel. Every candidate has to be considered.
    """
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), isolated_vllm: (True, True)},
    )

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert isolated_vllm in capsys.readouterr().out


def test_isolated_venv_is_probed_before_the_benchmark_interpreter(isolated_vllm, monkeypatch):
    """Mirror install_baremetal.sh, which switches the probe to the venv."""
    calls = _probe_per_interpreter(monkeypatch, {isolated_vllm: (True, True)})

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert calls[0][0] == isolated_vllm


def test_every_importable_candidate_cuda_still_fails(isolated_vllm, monkeypatch, capsys):
    """Scanning all candidates must not weaken the gate when all are wrong."""
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), isolated_vllm: (True, False)},
    )

    with pytest.raises(SystemExit) as excinfo:
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert excinfo.value.code == 2
    assert "NOT a ROCm build" in capsys.readouterr().err


def test_an_inconclusive_candidate_outweighs_a_refuted_one(isolated_vllm, monkeypatch, capsys):
    """Block only when every importable candidate is provably the wrong build."""
    _probe_per_interpreter(
        monkeypatch,
        {"/usr/bin/python3": (True, False), isolated_vllm: (True, None)},
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
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3").verdict is True

    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: _Proc(1))
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3").verdict is False

    def _boom(*_a, **_k):
        raise OSError("probe crashed")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3").verdict is None

    # Absent torch and a signal death are "cannot answer", not "wrong build":
    # calling either a CUDA wheel would be a wrong diagnosis.
    for rc in (3, -11):
        monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, _rc=rc, **_k: _Proc(_rc))
        assert preflight._probe_rocm_build("vllm", "/usr/bin/python3").verdict is None


def test_vllm_probe_accepts_only_rocm_marked_unspecified_build(monkeypatch):
    scripts = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, *_args, **_kwargs):
        scripts.append(cmd[-1])
        return _Proc()

    monkeypatch.setattr(preflight.subprocess, "run", _run)
    assert preflight._probe_rocm_build("vllm", "/usr/bin/python3").verdict is True
    script = scripts[-1]
    assert "UnspecifiedPlatform" in script
    assert "'rocm' in version" in script
    assert "unspecified and 'rocm' in version" in script


def _host_rocm_verdict() -> bool | None:
    """What this host says about its own torch, resolved out of process.

    Importing torch in the pytest process would put a broken ROCm stack in the
    session itself, where a signal death on ``import torch`` takes the whole run
    with it -- the product spawns the probe for exactly that reason. Written
    independently of the production script so the two can disagree: no output at
    all, from any cause, reads as "cannot say", which is its rc-3 semantics.
    """
    script = (
        "import json, sys\n"
        "try:\n"
        "    import torch\n"
        "except BaseException:\n"
        "    print('null'); sys.exit(0)\n"
        "print(json.dumps(bool(getattr(torch.version, 'hip', None))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    answer = (proc.stdout or "").strip()
    if proc.returncode != 0 or not answer:
        return None
    return json.loads(answer)


def test_the_real_probe_agrees_with_this_host():
    """Runs the probe for real, and derives the expectation from the host.

    The stubbed tri-state test covers the return codes but never spawns an
    interpreter, and the subprocess path is where absent torch was once read as
    a CUDA wheel. A fixed verdict here would only hold where torch happens to be
    missing: green on a CI runner, red on every ROCm host the product targets.
    """
    probe = preflight._probe_rocm_build("sglang", sys.executable)

    assert probe.verdict is _host_rocm_verdict()


# ---------------------------------------------------------------------------
# probe diagnostics
# ---------------------------------------------------------------------------
def _dying_rocm_probe(monkeypatch, stderr: str, *, returncode: int = -11) -> None:
    """Importable framework whose ROCm probe dies, leaving only stderr behind."""

    class _Proc:
        def __init__(self, rc: int, err: str = "") -> None:
            self.returncode = rc
            self.stdout = ""
            self.stderr = err

    def _run(cmd, *_args, **_kwargs):
        return _Proc(0) if "find_spec" in cmd[-1] else _Proc(returncode, stderr)

    monkeypatch.setattr(preflight.subprocess, "run", _run)


def test_inconclusive_warning_surfaces_the_probe_stderr_tail(monkeypatch, capsys):
    """A broken ROCm stack is diagnosable only from the stderr the probe drops."""
    noise = [f"noise-{i}" for i in range(40)]
    _dying_rocm_probe(monkeypatch, "\n".join([*noise, "ImportError: libamdhip64.so: cannot open"]))

    preflight._check_serving_framework(_args("sglang"), "/usr/bin/python3")

    out = capsys.readouterr().out
    assert "could not verify" in out
    assert "libamdhip64.so" in out
    assert "noise-0" not in out


def test_probe_stderr_tail_is_bounded():
    """A ROCm import traceback can be enormous; the warning must stay readable."""
    tail = preflight._probe_stderr_tail("\n".join(f"line-{i} " + "x" * 400 for i in range(500)))

    assert 0 < len(tail) <= 1000
    assert "line-499" in tail
    assert "line-0 " not in tail


def test_a_remote_server_is_pointed_at_benchmark_base_url(monkeypatch, capsys):
    """A server that lives elsewhere is configured, not waved through.

    BENCHMARK_BASE_URL is an exemption in this very function, so steering that
    user into disabling the check hides the supported path.
    """
    _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    err = capsys.readouterr().err
    assert "BENCHMARK_BASE_URL" in err
    assert err.index("BENCHMARK_BASE_URL") < err.index(_SKIP_ENV)
    assert "last resort" in err


@pytest.mark.parametrize(
    ("framework", "evidence"),
    [("vllm", "vllm platform"), ("sglang", "torch.version.hip")],
)
def test_the_rocm_verdict_names_its_evidence(monkeypatch, capsys, framework, evidence):
    """Only vLLM reports its own platform; elsewhere torch's tag is all there is.

    Claiming the framework itself is a ROCm build would wave a CUDA sglang
    sitting beside a ROCm torch straight through.
    """
    _probe_result(monkeypatch, importable=True, rocm=True)

    preflight._check_serving_framework(_args(framework), "/usr/bin/python3")

    assert evidence in capsys.readouterr().out


def test_a_refuted_non_vllm_build_blames_torch(monkeypatch, capsys):
    """For sglang the refuted thing is torch, so the error must not overclaim."""
    _probe_result(monkeypatch, importable=True, rocm=False)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("sglang"), "/usr/bin/python3")

    err = capsys.readouterr().err
    assert "NOT a ROCm build" in err
    assert "torch.version.hip" in err


def _timing_out_probe(monkeypatch, *, importable: bool) -> list[float]:
    """Record every probe timeout; time out on the probe under test."""
    budget: list[float] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, *_args, **kwargs):
        timeout = kwargs.get("timeout") or 0
        budget.append(timeout)
        if importable and "find_spec" in cmd[-1]:
            return _Proc()
        raise preflight.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(preflight.subprocess, "run", _run)
    return budget


@pytest.mark.parametrize("importable", [False, True])
def test_a_probe_timeout_stops_the_interpreter_scan(monkeypatch, capsys, importable):
    """Paying a timeout per candidate turned preflight into a 12-minute wait.

    A host slow enough to blow one budget will blow the next two as well, and a
    timeout proves nothing, so the scan stops and the run proceeds unverified.
    """

    budget = _timing_out_probe(monkeypatch, importable=importable)
    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert sum(budget) <= 150, f"worst-case probe budget too large: {budget}"
    assert "timed out" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _in_container signals
# ---------------------------------------------------------------------------
# A systemd host puts PID 1 in a named scope; a container with a private cgroup
# namespace sees the namespace root instead.
_HOST_CGROUP_V2 = "0::/init.scope\n"
_CONTAINER_CGROUP_V2 = "0::/\n"


def _fake_container_fs(monkeypatch, *, present=(), cgroup=_HOST_CGROUP_V2):
    """Fake every file _in_container reads; only ``present`` paths exist.

    ``cgroup=None`` makes ``/proc/1/cgroup`` unreadable.
    """

    class _Node:
        def __init__(self, path):
            self._path = str(path)

        def exists(self):
            return self._path in present

        def read_text(self, **_kwargs):
            if self._path == "/proc/1/cgroup" and cgroup is not None:
                return cgroup
            raise OSError(f"unreadable: {self._path}")

    monkeypatch.setattr(preflight, "Path", _Node)
    monkeypatch.setattr(provenance, "_read_first_line", lambda _p: "")


def test_in_container_detects_a_cgroup_v2_namespace_root(monkeypatch):
    """Under cgroup v2 the runtime name is gone: /proc/1/cgroup is just "0::/".

    Missing it tells a container user to start a container, the reversal of the
    advice #1141 exists to fix.
    """
    _fake_container_fs(monkeypatch, cgroup=_CONTAINER_CGROUP_V2)

    assert preflight._in_container() is True


def test_in_container_detects_podman(monkeypatch):
    """podman writes /run/.containerenv, never /.dockerenv."""
    _fake_container_fs(monkeypatch, present=("/run/.containerenv",), cgroup=_CONTAINER_CGROUP_V2)

    assert preflight._in_container() is True


def test_in_container_detects_a_kubernetes_pod(monkeypatch):
    """Every pod gets the API service env injected, whatever the cgroup shape."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    _fake_container_fs(monkeypatch)

    assert preflight._in_container() is True


def test_in_container_trusts_an_on_disk_image_marker(monkeypatch):
    """A projected/baked image marker file only exists inside the image."""
    _fake_container_fs(monkeypatch)
    monkeypatch.setattr(provenance, "_read_first_line", lambda p: "rocm-image" if "podinfo" in str(p) else "")

    assert preflight._in_container() is True


def test_host_image_env_is_not_container_proof(monkeypatch):
    """The demo skills export HYPERLOOM_IMAGE on the *host* to pick an image."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "rocm-image")
    _fake_container_fs(monkeypatch)

    assert preflight._in_container() is False


def test_in_container_is_false_on_a_cgroup_v2_host(monkeypatch):
    """Regression guard: the dev host is cgroup v2 and is not a container."""
    _fake_container_fs(monkeypatch)

    assert preflight._in_container() is False


def test_in_container_prefers_container_when_no_signal_is_readable(monkeypatch):
    """A false negative gives wrong advice; a false positive only less specific."""
    _fake_container_fs(monkeypatch, cgroup=None)

    assert preflight._in_container() is True


_SETUP_INSTALLER = "inference_optimizer/assets/install_baremetal.sh"


def test_a_framework_setup_cannot_install_is_not_blocked(monkeypatch, capsys):
    """atom is a serving framework with no installer path.

    install_baremetal.sh takes only none|sglang|vllm and exits 2 on anything
    else, and never probes atom at all, so blocking the run and naming
    ``--install-framework atom`` walks the reader into a second wall.
    """
    _probe_result(monkeypatch, importable=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    preflight._check_serving_framework(_args("atom"), "/usr/bin/python3")

    out = capsys.readouterr().out
    assert "atom" in out
    assert "--install-framework atom" not in out


def test_the_installable_set_matches_the_installer():
    """Two lists that must agree, in different languages, with nothing else
    tying them together."""
    from pathlib import Path

    import hyperloom

    source = (Path(hyperloom.__file__).parent / _SETUP_INSTALLER).read_text(encoding="utf-8")
    accepted = re.search(r"^\s*(none\|[a-z|]+)\)\s*;;", source, re.MULTILINE)
    assert accepted, "could not find the --install-framework case arm"
    declared = {value for value in accepted.group(1).split("|") if value != "none"}

    assert declared == set(preflight._SETUP_INSTALLABLE_FRAMEWORKS), (
        f"installer accepts {declared}, preflight believes {set(preflight._SETUP_INSTALLABLE_FRAMEWORKS)}"
    )


def test_an_uninstallable_framework_is_not_blocked_for_a_cuda_build(monkeypatch, capsys):
    """The refuted branch is the other way into the same dead end.

    A CUDA torch with an atom checkout on PYTHONPATH is uncommon, but it lands on
    exactly the remedy that cannot work, which is what the exemption exists for.
    """
    _probe_result(monkeypatch, importable=True, rocm=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    preflight._check_serving_framework(_args("atom"), "/usr/bin/python3")

    combined = capsys.readouterr()
    assert "--install-framework atom" not in combined.out + combined.err


# ---------------------------------------------------------------------------
# Wiring: nothing above proves _preflight still calls the gate
# ---------------------------------------------------------------------------
def test_preflight_still_invokes_the_gate():
    """Every other test calls the gate directly, so deleting the one line that
    reaches it from _preflight would leave them all green."""
    import ast
    from pathlib import Path as _Path

    source = _Path(preflight.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_preflight")
    called = {
        node.func.id for node in ast.walk(target) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_check_serving_framework" in called


@pytest.mark.parametrize("framework", sorted(preflight._SETUP_INSTALLABLE_FRAMEWORKS))
def test_the_remedy_matches_the_documented_setup_invocation(framework):
    """Printing a command promises it runs.

    The skill's form carries PYTHONPATH and --yes, and vLLM needs the isolated
    env; a remedy missing any of them fails or hangs on a prompt -- the same
    shape of dead end as naming a framework setup cannot install.
    """
    from pathlib import Path as _Path

    import hyperloom

    skill = (_Path(hyperloom.__file__).parent / "skills/hyperloom-setup/SKILL.md").read_text(encoding="utf-8")
    documented = [
        line.strip()
        for line in skill.splitlines()
        if "inference_optimizer.setup" in line and f"--install-framework {framework}" in line
    ]
    assert documented, f"no documented setup line for {framework}"

    assert preflight._setup_install_command(framework) in documented


# --- the resolved-interpreter publish contract -----------------------------


def test_a_resolved_interpreter_is_published_with_its_framework(monkeypatch):
    """provenance reads the pair; an unlabelled interpreter is unusable.

    The scan answers for one framework, and ``sglang`` is the default, so the
    name has to travel with the path or a vLLM lookup would read an SGLang
    interpreter as its own answer.
    """
    _probe_result(monkeypatch, importable=True)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert os.environ[RESOLVED_FRAMEWORK_PYTHON_ENV] == "/usr/bin/python3"
    assert os.environ[RESOLVED_FRAMEWORK_ENV] == "vllm"


def test_a_refuted_build_is_not_published(monkeypatch):
    """A candidate proven to be the wrong build must not become the answer.

    The check exits on a refuted build; the point here is that nothing was
    published on the way out.
    """
    _probe_result(monkeypatch, importable=True, rocm=False)
    monkeypatch.setattr(preflight, "_in_container", lambda: False)

    with pytest.raises(SystemExit):
        preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert RESOLVED_FRAMEWORK_PYTHON_ENV not in os.environ
    assert RESOLVED_FRAMEWORK_ENV not in os.environ


def test_a_rocm_probe_timeout_still_publishes(monkeypatch):
    """The timeout is in the ROCm verdict, not in importability.

    ``_resolve_framework_build`` proves the candidate importable before probing
    the build at all, and the check keeps serving with it after a warning.
    Withholding it here would send provenance back to this process and record
    the orchestrator's own version for a run served by the isolated venv.
    """
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(preflight.subprocess, "run", lambda cmd, *a, **k: (calls.append(list(cmd)), _Proc())[1])
    monkeypatch.setattr(
        preflight,
        "_probe_rocm_build",
        lambda _fw, _py: preflight._Probe(None, timed_out=True),
    )

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert os.environ.get(RESOLVED_FRAMEWORK_PYTHON_ENV) == "/usr/bin/python3"
    assert os.environ.get(RESOLVED_FRAMEWORK_ENV) == "vllm"


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(lambda mp: mp.setenv(_SKIP_ENV, "1"), id="skip_env"),
        pytest.param(lambda mp: mp.setenv("BENCHMARK_BASE_URL", "http://serving-host:8888"), id="remote_url"),
        pytest.param(
            lambda mp: (
                mp.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://claw-rayjob:8000"),
                mp.setenv("INFERENCE_OPTIMIZER_NODES", "2"),
            ),
            id="external_multi_node",
        ),
    ],
)
def test_an_exempt_path_publishes_nothing(monkeypatch, setup):
    """Serving is not local on these paths, so there is no resolution to publish."""
    setup(monkeypatch)
    _refuse_probe(monkeypatch)

    preflight._check_serving_framework(_args("vllm"), "/usr/bin/python3")

    assert RESOLVED_FRAMEWORK_PYTHON_ENV not in os.environ
    assert RESOLVED_FRAMEWORK_ENV not in os.environ


def test_a_non_venv_interpreter_does_not_force_unknown(monkeypatch, tmp_path):
    """A system prefix keeps packages in ``dist-packages``, not ``site-packages``.

    Deriving ``<prefix>/lib/python*/site-packages`` finds nothing there, and
    treating that as an authoritative empty answer would record "unknown" for a
    framework this process can see -- the failure mode this whole path exists to
    remove. Bare-metal Debian/Ubuntu without a venv is a supported layout.

    The prefix is built here rather than probing the host's own ``/usr``: RHEL
    and several ROCm images do keep ``lib/python*/site-packages`` under it, so
    asserting against the real one would pass or fail by runner.
    """
    (tmp_path / "lib" / "python3.12" / "dist-packages").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    interpreter = tmp_path / "bin" / "python3"
    interpreter.touch()

    _probe_result(monkeypatch, importable=True)
    preflight._check_serving_framework(_args("vllm"), str(interpreter))

    assert provenance._framework_site_packages(os.environ, "vllm") is None
