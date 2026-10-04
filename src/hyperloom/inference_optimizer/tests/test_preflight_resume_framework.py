# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight checks the resumed session's framework, not the CLI default.

Preflight runs before the resume block loads the session. A ``--resume-from`` that does not re-pass ``--framework``
used to install and probe the default framework (sglang) for a session created with ``--framework vllm``, and failed
with "serving framework 'sglang' is not importable" on a host that only carries vllm.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.cli import preflight


@pytest.fixture(autouse=True)
def _restore_environment(tmp_path):
    snapshot = dict(os.environ)
    home = tmp_path / "home"
    home.mkdir()
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _session(tmp_path: Path, **state: object) -> Path:
    session_dir = tmp_path / "workspace" / "model" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return session_dir


def _args(**kw: object) -> SimpleNamespace:
    return SimpleNamespace(**{"framework": None, "resume_from": "", "no_kernel": False, **kw})


class _ReachedServingCheck(Exception):
    pass


def _run_preflight_to_serving_check(monkeypatch, args) -> list[tuple[str, str]]:
    """Drive the real ``_preflight`` through the framework-deps and serving-framework steps.

    Every other install step is recorded and skipped; the two framework steps run their real code down to the point
    where they name a framework (the deps installer and the importability probe), which is recorded.
    """
    seen: list[tuple[str, str]] = []

    class _StopDeps(Exception):
        pass

    def fake_ensure(framework, *, python_exe, pip_extra):
        seen.append(("framework_deps", framework))
        raise _StopDeps

    def fake_resolve_build(framework, interpreters):
        seen.append(("check_serving_framework", framework))
        raise _ReachedServingCheck

    def run_install_step(event, *, step_id, category, action, **_kw):
        if step_id == "framework_deps":
            # The deps step only records which framework it was asked to install; stop it before any pip run.
            with contextlib.suppress(_StopDeps):
                action()
            return {}
        if step_id == "check_serving_framework":
            return action()
        return {}

    from hyperloom.inference_optimizer import framework_deps

    monkeypatch.setattr(framework_deps, "ensure", fake_ensure)
    monkeypatch.setattr(preflight, "_resolve_framework_build", fake_resolve_build)
    monkeypatch.setattr(preflight, "_run_install_step", run_install_step)
    monkeypatch.setattr(preflight, "_resolve_llm_endpoints", lambda: ("", ""))
    monkeypatch.setattr(preflight, "_normalize_hip_visible_devices", lambda: None)
    monkeypatch.delenv(preflight.SKIP_FRAMEWORK_CHECK_ENV, raising=False)
    monkeypatch.delenv("BENCHMARK_BASE_URL", raising=False)
    with pytest.raises(_ReachedServingCheck):
        preflight._preflight(args)
    return seen


@pytest.mark.parametrize("environment_framework", [None, "sglang"])
def test_resume_without_framework_checks_the_sessions_framework(monkeypatch, tmp_path, environment_framework):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    if environment_framework is not None:
        monkeypatch.setenv("FRAMEWORK", environment_framework)
    args = _args(resume_from=str(_session(tmp_path, framework="vllm")))

    seen = _run_preflight_to_serving_check(monkeypatch, args)

    assert seen == [("framework_deps", "vllm"), ("check_serving_framework", "vllm")]
    assert args.framework == "vllm"
    assert os.environ["FRAMEWORK"] == "vllm"


def test_resume_with_the_matching_framework_checks_it(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    args = _args(framework="vllm", resume_from=str(_session(tmp_path, framework="vllm")))

    seen = _run_preflight_to_serving_check(monkeypatch, args)

    assert seen == [("framework_deps", "vllm"), ("check_serving_framework", "vllm")]
    assert args.framework == "vllm"


def test_resume_with_a_different_framework_is_refused_before_any_check(monkeypatch, tmp_path, capsys):
    def began_checking(_args):
        raise AssertionError("preflight started its checks for a resume it should refuse")

    monkeypatch.setattr(preflight, "_begin_install_event", began_checking)
    args = _args(framework="sglang", resume_from=str(_session(tmp_path, framework="vllm")))

    with pytest.raises(SystemExit) as excinfo:
        preflight._preflight(args)

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "created with --framework vllm" in err
    assert "passes --framework sglang" in err
    assert args.framework == "sglang"


def test_fresh_launch_keeps_the_default_framework(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    args = _args()

    seen = _run_preflight_to_serving_check(monkeypatch, args)

    assert seen == [("framework_deps", "sglang"), ("check_serving_framework", "sglang")]
    assert args.framework is None


def test_a_session_without_a_persisted_framework_keeps_the_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAMEWORK", raising=False)
    args = _args(framework="atom", resume_from=str(_session(tmp_path, framework="")))

    preflight._pin_resumed_session_args(args)

    assert args.framework == "atom"


def test_an_unreadable_session_state_leaves_the_flags_alone(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "state.json").write_text("{not json", encoding="utf-8")
    args = _args(framework="vllm", resume_from=str(session_dir))

    preflight._pin_resumed_session_args(args)

    assert args.framework == "vllm"
    assert args.no_kernel is False


@pytest.mark.parametrize(("kernel_enabled", "expected_no_kernel"), [(False, True), (True, False)])
def test_resume_carries_the_persisted_kernel_toggle(tmp_path, kernel_enabled, expected_no_kernel):
    args = _args(resume_from=str(_session(tmp_path, framework="vllm", kernel_enabled=kernel_enabled)))

    preflight._pin_resumed_session_args(args)

    assert args.no_kernel is expected_no_kernel
    assert preflight._tracelens_required_at_preflight(args.no_kernel, False) is (not expected_no_kernel)
