# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight lm_eval accuracy-gate dependency ensure."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.cli import preflight


class _FakeRun:
    """Stand-in for ``subprocess.run`` driving one probe subprocess per module.

    ``missing`` names the modules whose ``import`` probe fails; ``dead_interpreter``
    makes even the ``pass`` liveness probe fail.
    """

    def __init__(
        self,
        missing,
        *,
        dead_interpreter: bool = False,
        install_rc: int = 0,
        failing_specs=(),
        versions=None,
    ):
        self.missing = list(missing)
        self.dead_interpreter = dead_interpreter
        self.install_rc = install_rc
        self.failing_specs = list(failing_specs)
        self.versions = {"torch": "2.6.0", "pandas": "2.2.3", "numpy": "1.26.4", "triton": "3.2.0"}
        if versions is not None:
            self.versions = dict(versions)
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
        if cmd[1:2] == ["-c"] and "importlib.metadata" in cmd[2]:  # installed-version probe
            name = cmd[2].rsplit("'", 2)[-2]
            if name in self.versions:
                r.stdout = f"{self.versions[name]}\n"
            else:
                r.returncode = 1
            return r
        if cmd[1:2] == ["-c"]:  # per-module import probe
            module = cmd[2].removeprefix("import ")
            r.returncode = 1 if module in self.missing else 0
            return r
        if any(spec in cmd for spec in self.failing_specs):
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd)
            r.returncode = 1
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
        return [
            c[2].removeprefix("import ")
            for c in self.calls
            if c[1:2] == ["-c"] and c[2] != "pass" and "importlib.metadata" not in c[2]
        ]


def _patch(monkeypatch, runner):
    monkeypatch.setattr(preflight.subprocess, "run", runner)
    return runner


@pytest.fixture(autouse=True)
def _multi_node(monkeypatch):
    """The ensure is multi-node only; single-node coverage sets this explicitly."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")


def test_lm_eval_installed_when_missing_and_eval_enabled(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)  # default => enabled
    runner = _patch(monkeypatch, _FakeRun(["lm_eval", "tenacity"]))

    preflight._ensure_lm_eval_dep("py", ["--break-system-packages"])

    assert len(runner.installs) == 1
    # Pinned to the commit InferenceX force-reinstalls on the single-node path,
    # so both paths measure accuracy with the same harness.
    assert preflight._LM_EVAL_PINNED_SPECS[0][1] in runner.installs[0]
    assert preflight._LM_EVAL_PINNED_REF in runner.installs[0][-1]


def _constraint_lines(install_cmd: list[str]) -> list[str]:
    """Read back the constraints file a pip invocation was handed."""
    assert "-c" in install_cmd, f"install ran unconstrained: {install_cmd}"
    return Path(install_cmd[install_cmd.index("-c") + 1]).read_text(encoding="utf-8").split()


def test_lm_eval_install_cannot_move_the_packages_install_sh_settled(monkeypatch):
    """This install runs after install.sh's deliberately-last pandas pin.

    install.sh orders ``ensure_rocprof_compute`` after every pip step precisely
    so "no later pip install can re-pull pandas>=3"; pandas>=3 makes
    rocprof-compute drop every counter and forge degrade to PMC with no
    roofline. This install happens at optimize time -- after that last word, in
    the same interpreter -- and its closure reaches pandas through datasets and
    torch directly, where a PyPI torch would be a CUDA build on a ROCm box.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["lm_eval"]))

    preflight._ensure_lm_eval_dep("py", [])

    pins = _constraint_lines(runner.installs[0])
    assert "pandas==2.2.3" in pins
    assert "torch==2.6.0" in pins


def test_a_missing_extra_is_installed_under_the_same_constraints(monkeypatch):
    """The second install path resolves a closure too, so it carries them too."""
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["tenacity"]))

    preflight._ensure_lm_eval_dep("py", [])

    assert "pandas==2.2.3" in _constraint_lines(runner.installs[0])


def test_unreadable_versions_warn_and_install_anyway(monkeypatch, capsys):
    """A pin that cannot be read must not block the accuracy gate outright."""
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["lm_eval"], versions={}))

    preflight._ensure_lm_eval_dep("py", [])

    assert "-c" not in runner.installs[0]
    assert "unconstrained" in capsys.readouterr().out


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


def test_pinned_install_falls_back_to_archive_without_git(monkeypatch):
    """No git binary must not fail the gate: the source archive is the fallback.

    ``check=True`` on the git spec would turn a gitless sandbox into an aborted
    preflight, so only the last spec may raise.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    git_spec, archive_spec = (s for _, s in preflight._LM_EVAL_PINNED_SPECS)
    runner = _patch(monkeypatch, _FakeRun(["lm_eval", "tenacity"], failing_specs=[git_spec]))

    preflight._ensure_lm_eval_dep("py", [])

    assert [i[-1] for i in runner.installs] == [git_spec, archive_spec]


def test_image_pinned_lm_eval_is_not_replaced(monkeypatch):
    """Only the absent extra is installed when the image already ships lm_eval.

    Force-reinstalling the pin here would swap out a version the image chose on
    purpose, so the pinned spec must stay out of this install.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    runner = _patch(monkeypatch, _FakeRun(["tenacity"]))  # lm_eval imports fine

    preflight._ensure_lm_eval_dep("py", [])

    assert len(runner.installs) == 1
    assert runner.installs[0][-1] == "tenacity"
    assert preflight._LM_EVAL_PINNED_REF not in " ".join(runner.installs[0])


def test_single_node_is_left_to_inferencex(monkeypatch):
    """Single-node must not even probe: preflight never touched lm_eval there.

    ``run_eval`` -> InferenceX ``run_lm_eval`` installs the harness on first use
    and force-reinstalls its own pinned commit over anything already present, so
    installing ahead of it cannot change the outcome -- it can only add a way for
    a previously working run to die on ``check=True``.
    """
    monkeypatch.delenv("RUN_EVAL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")

    def _boom(*_a, **_k):
        raise AssertionError("single-node must not probe or install lm_eval")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    preflight._ensure_lm_eval_dep("py", [])


def test_lm_eval_skipped_when_run_eval_disabled(monkeypatch):
    monkeypatch.setenv("RUN_EVAL", "false")

    def _boom(*_a, **_k):
        raise AssertionError("must not probe/install when RUN_EVAL is disabled")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    preflight._ensure_lm_eval_dep("py", [])


def test_lm_eval_skipped_under_no_eval(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("must not probe/install under --no-eval")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    preflight._ensure_lm_eval_dep("py", [], eval_disabled=True)


# --- _resolved_eval_disabled: preflight runs before the resume block --------
def _args(**kw):
    return SimpleNamespace(**{"no_eval": False, "resume": False, "resume_from": "", **kw})


def test_resolved_eval_disabled_reads_the_flag():
    assert preflight._resolved_eval_disabled(_args(no_eval=True)) is True
    assert preflight._resolved_eval_disabled(_args()) is False


def test_resolved_eval_disabled_reads_the_resumed_session(tmp_path):
    (tmp_path / "state.json").write_text('{"eval_disabled": true}', encoding="utf-8")
    assert preflight._resolved_eval_disabled(_args(resume=True, resume_from=str(tmp_path))) is True


def test_resolved_eval_disabled_without_a_readable_state(tmp_path):
    assert preflight._resolved_eval_disabled(_args(resume=True, resume_from=str(tmp_path))) is False


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
    # tenacity still judged present: only lm_eval is installed
    assert preflight._LM_EVAL_PINNED_SPECS[0][1] in runner.installs[0]


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
