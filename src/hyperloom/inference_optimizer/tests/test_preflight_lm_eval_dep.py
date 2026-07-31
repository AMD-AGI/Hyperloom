# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight lm_eval accuracy-gate dependency ensure."""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.inference_optimizer.cli import preflight


class _FakeRun:
    """Stand-in for ``subprocess.run`` driving one probe subprocess per module.

    ``missing`` names the modules whose ``import`` probe fails; ``dead_interpreter``
    makes even the ``pass`` liveness probe fail.
    """

    def __init__(self, missing, *, dead_interpreter: bool = False, install_rc: int = 0):
        self.missing = list(missing)
        self.dead_interpreter = dead_interpreter
        self.install_rc = install_rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        cmd = list(cmd)
        self.calls.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        r = _R()
        if cmd[1:3] == ["-c", "pass"]:  # liveness probe
            r.returncode = 1 if self.dead_interpreter else 0
            return r
        if cmd[1:2] == ["-c"]:  # per-module import probe
            module = cmd[2].removeprefix("import ")
            r.returncode = 1 if module in self.missing else 0
            return r
        r.returncode = self.install_rc
        if self.install_rc and kwargs.get("check"):
            raise subprocess.CalledProcessError(self.install_rc, cmd)
        return r

    @property
    def installs(self) -> list[list[str]]:
        return [c for c in self.calls if c[1:4] == ["-m", "pip", "install"]]

    @property
    def probed_modules(self) -> list[str]:
        return [c[2].removeprefix("import ") for c in self.calls if c[1:2] == ["-c"] and c[2] != "pass"]


def _patch(monkeypatch, runner):
    monkeypatch.setattr(preflight.subprocess, "run", runner)
    return runner


def test_lm_eval_installed_when_missing_and_eval_enabled(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)  # default => enabled
    runner = _patch(monkeypatch, _FakeRun(["lm_eval", "tenacity"]))

    preflight._ensure_lm_eval_dep("py", ["--break-system-packages"])

    assert len(runner.installs) == 1
    assert "lm_eval[api]" in runner.installs[0]


def test_lm_eval_skipped_when_present(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun([]))

    preflight._ensure_lm_eval_dep("py", [])

    assert runner.installs == []  # probe only, no install


def test_image_provided_lm_eval_is_never_reinstalled_for_a_missing_extra(monkeypatch, capsys):
    """An image that ships lm_eval keeps its own build; only the extra is added.

    Probing ``lm_eval`` and ``tenacity`` as one import made these two states
    indistinguishable, so a missing extra triggered ``pip install lm_eval[api]``
    and let pip resolve a different lm_eval over the version the image pinned.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["tenacity"]))

    preflight._ensure_lm_eval_dep("py", [])

    assert len(runner.installs) == 1
    install = runner.installs[0]
    assert "tenacity" in install
    assert not any("lm_eval" in arg for arg in install)
    assert "installing tenacity" in capsys.readouterr().out


def test_lm_eval_skipped_when_run_eval_disabled(monkeypatch):
    monkeypatch.setenv("RUN_EVAL", "false")

    def _boom(*_a, **_k):
        raise AssertionError("must not probe/install when RUN_EVAL is disabled")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    preflight._ensure_lm_eval_dep("py", [])


def test_unprobeable_interpreter_is_left_untouched(monkeypatch, capsys):
    """A probe that cannot run proves nothing, so nothing is installed over it.

    Guessing here would reinstall on top of an lm_eval the image may already
    ship. An interpreter that cannot run ``python -c`` breaks the benchmark far
    more visibly than a missing accuracy gate, so this warns instead of raising.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["lm_eval", "tenacity"], dead_interpreter=True))

    preflight._ensure_lm_eval_dep("py", [])

    assert runner.installs == []
    assert runner.probed_modules == []  # gave up before importing anything
    assert "cannot run the lm_eval probe" in capsys.readouterr().out


def test_unusable_interpreter_path_does_not_crash_preflight(monkeypatch, capsys):
    """A missing interpreter raises from subprocess rather than returning a code.

    Preflight must not die on it: the accuracy gate is not worth aborting the
    launch over, and absence is still unproven, so nothing is installed either.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    calls: list[list[str]] = []

    def _enoent(cmd, **_kwargs):
        calls.append(list(cmd))
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(preflight.subprocess, "run", _enoent)

    preflight._ensure_lm_eval_dep("/nonexistent/python", [])

    assert not any(c[1:4] == ["-m", "pip", "install"] for c in calls)
    assert "cannot run the lm_eval probe" in capsys.readouterr().out


def test_each_module_is_probed_in_its_own_subprocess(monkeypatch):
    """A hard crash importing one module must not void the verdict on the others.

    ``import lm_eval`` pulls in torch, which a broken ROCm install can kill by
    signal rather than by exception. Probing both modules in one interpreter let
    that take tenacity's result down with it, and the whole ensure then bailed
    out as unprobeable.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["lm_eval"]))  # lm_eval import dies

    preflight._ensure_lm_eval_dep("py", [])

    assert runner.probed_modules == ["lm_eval", "tenacity"]
    assert len(runner.installs) == 1
    assert "lm_eval[api]" in runner.installs[0]  # tenacity still judged present


def test_lm_eval_install_failure_aborts_preflight(monkeypatch):
    """A failed pip install must abort preflight instead of being swallowed.

    The whole point of this ensure is that ``lm_eval`` exists before the run
    starts. Ignoring the install exit code let a broken network, resolver or
    permission land back on ``No module named lm_eval`` ->
    ``baseline_accuracy_failed`` hours later, with the pip diagnostics gone --
    exactly the failure this function was added to remove.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    _patch(monkeypatch, _FakeRun(["lm_eval", "tenacity"], install_rc=1))

    with pytest.raises(subprocess.CalledProcessError):
        preflight._ensure_lm_eval_dep("py", [])
