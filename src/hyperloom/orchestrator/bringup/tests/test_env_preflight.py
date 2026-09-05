# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The five faults that belong to the host, and the abstentions around them."""

from __future__ import annotations

import json
import socket
import subprocess

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.bringup import env_preflight as ep


def _probe_returning(payload: dict | None, *, stdout: str = "", returncode: int = 0):
    """Return a probe shim that answers every call with one payload.

    Args:
        payload: JSON object the probe program "printed"; ``None`` to print
            ``stdout`` verbatim instead.
        stdout: Raw stdout when ``payload`` is ``None``.
        returncode: Exit code the shim reports.

    Returns:
        A callable with the probe signature.
    """

    def _probe(argv, env, timeout_sec):
        body = json.dumps(payload) + "\n" if payload is not None else stdout
        return subprocess.CompletedProcess(list(argv), returncode, stdout=body, stderr="")

    return _probe


@pytest.fixture
def serving_python():
    """A launch env that pins the interpreter the server would run in."""
    return {"HYPERLOOM_FRAMEWORK_PYTHON": "/opt/serve/bin/python"}


def test_a_checkpoint_path_that_does_not_resolve_is_a_fault(tmp_path, serving_python):
    """An absolute weights path with nothing behind it stops the round."""
    verdict = ep.check_environment(
        framework="vllm",
        model=str(tmp_path / "absent-checkpoint"),
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"found": True, "exc": "", "missing": "", "detail": ""}),
    )
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.CHECKPOINT_UNRESOLVED
    assert verdict.terminal


def test_a_repository_id_is_not_judged_as_a_path(serving_python):
    """A hub id names no local path, so nothing about this host refutes it."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"found": True, "exc": "", "missing": "", "detail": ""}),
    )
    assert verdict.status == ep.OK


def test_a_port_already_bound_is_a_fault(serving_python):
    """Something answering on the serving port means this round cannot bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        verdict = ep.check_environment(
            framework="vllm",
            model="org/Model-8B",
            port=port,
            launch_env=serving_python,
            probe=_probe_returning({"found": True, "exc": "", "missing": "", "detail": ""}),
        )
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.PORT_ALREADY_BOUND


def test_a_framework_the_serving_interpreter_cannot_resolve_is_a_fault(serving_python):
    """``find_spec`` finding nothing is the whole of the evidence needed."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"found": False, "exc": "", "missing": "", "detail": ""}),
    )
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.FRAMEWORK_NOT_INSTALLED


def test_a_missing_dependency_is_named_from_the_exception_not_the_message(serving_python):
    """The subject comes off ``ModuleNotFoundError.name``, not the wording."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning(
            {
                "found": True,
                "exc": "ModuleNotFoundError",
                "missing": "flashinfer",
                # Deliberately says nothing a matcher could key on.
                "detail": "voluntary interruption",
            }
        ),
    )
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.EXTENSION_MISSING
    assert verdict.subject == "flashinfer"


def test_an_extension_with_no_build_for_this_platform_is_told_apart_by_class(serving_python):
    """An ``ImportError`` that is not a ``ModuleNotFoundError`` is an unloadable build."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"found": True, "exc": "ImportError", "missing": "", "detail": "undefined symbol"}),
    )
    assert verdict.status == ep.FAULT
    assert verdict.fault == ep.EXTENSION_UNBUILT


def test_an_import_error_that_is_not_about_the_host_is_left_to_the_launch(serving_python):
    """A framework that raises for its own reasons is a boot failure, not a fault here."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"found": True, "exc": "ValueError", "missing": "", "detail": "bad config"}),
    )
    assert verdict.status == ep.OK


def test_a_probe_that_answers_with_something_else_is_unavailable_not_a_fault(serving_python):
    """An unrecognised payload cannot be read as "the framework is absent"."""
    verdict = ep.check_environment(
        framework="vllm",
        model="org/Model-8B",
        port=0,
        launch_env=serving_python,
        probe=_probe_returning({"status": "ok", "unknown": []}),
    )
    assert verdict.status == ep.UNAVAILABLE
    assert not verdict.terminal


def test_no_serving_interpreter_is_unavailable_not_a_fault(monkeypatch):
    """A check that could not be made is never evidence of a healthy host."""
    monkeypatch.setattr(ep, "resolve_serving_interpreter", lambda _fw, _env: "")
    verdict = ep.check_environment(framework="vllm", model="org/Model-8B", port=0, launch_env={})
    assert verdict.status == ep.UNAVAILABLE
    assert not verdict.terminal


def test_the_observation_carries_the_fault_and_the_stage_the_launch_would_die_at():
    """A fault records like any other bring-up observation, and says it is one."""
    verdict = ep.EnvVerdict(status=ep.FAULT, fault=ep.EXTENSION_UNBUILT, detail="undefined symbol")
    observation = ep.env_fault_observation(verdict)
    assert observation.stage_failed == LadderStage.IMPORT
    assert observation.env_fault == ep.EXTENSION_UNBUILT
    assert ep.is_env_fault(observation)
    assert not observation.booted


def test_a_boot_failure_the_classifier_placed_is_not_an_environment_fault():
    """Only an observation carrying ``env_fault`` ends the run as infrastructure."""
    from hyperloom.orchestrator.bringup.ladder import classify

    observation = classify(
        server_log="ValueError: Architectures ['XForCausalLM'] are not supported for now.",
        server_elapsed_sec=1.0,
    )
    assert not ep.is_env_fault(observation)


def test_a_host_resource_shortfall_is_not_one_of_the_terminal_faults():
    """A configuration that asks for more than the host has is still the loop's to fix."""
    from hyperloom.agents.framework.enablement import RESOURCE_CONSTRAINT
    from hyperloom.orchestrator.bringup.ladder import classify

    observation = classify(
        server_log="RuntimeError: HIP out of memory. Tried to allocate 4.00 GiB",
        server_elapsed_sec=1.0,
    )
    assert observation.env_fault == RESOURCE_CONSTRAINT
    assert not ep.is_env_fault(observation)
