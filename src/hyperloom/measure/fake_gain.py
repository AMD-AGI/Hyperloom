# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public names for main's refusals of a fake gain, so the multi-branch toolbox refuses the same requests.

Main refuses three gains that a measurement can show without the candidate being better: a context-copying
speculative draft on a synthetic benchmark, a change of a draft whose acceptance length is simulated, and an AgentX
baseline served with the prefix cache off. Its flow does that in the explore and baseline executors; a test launched
through ``hyperloom.measure`` never reaches them. This module re-exports the checks without copying their logic.

``FAKE_GAINS`` is the catalog of the known fake gains. The rule: each newly found fake gain lands as a check, a row
here and a test together; the test sends a minimal example through the real enforcement point named in the row.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.launch_log_evidence import observed_server_config_from_log
from hyperloom.contracts.recipe import LaunchConfig
from hyperloom.orchestrator.actions.executors import (
    PREFIX_CACHE_OFF_ERROR_CLASS,
    CONTEXT_COPY_REFUSAL,
    RoundFolder,
    SpeculativeDecodingPinnedError,
    context_copy_refusal,
    is_round_folder,
    refuse_cacheless_agentx_baseline,
    refuse_unpinned_speculative_change,
)

#: The ``RequestRejected`` code a candidate is refused under when it changes the draft or simulated acceptance its
#: parent pins; main's grid records the same refusal as a candidate ``serve_flag`` failure.
SPECULATIVE_DECODING_PINNED = "speculative_decoding_pinned"


class RoundUnreadable(ValueError):
    """An inference_suite round whose report or launch breaks its contract: no measurement to judge."""


__all__ = [
    "CONTEXT_COPY_REFUSAL",
    "PREFIX_CACHE_OFF_ERROR_CLASS",
    "FAKE_GAINS",
    "FakeGain",
    "RoundUnreadable",
    "SPECULATIVE_DECODING_PINNED",
    "baseline_prefix_cache_refusal",
    "launch_context_copy_refusal",
    "pinned_speculative_refusal",
]


def _argv(launch: LaunchConfig) -> tuple[str, dict[str, Any]]:
    return launch.server_args, dict(launch.envs)


def launch_context_copy_refusal(base: LaunchConfig, candidate: LaunchConfig) -> str:
    """Why ``candidate`` may not run on a synthetic (InferenceX) benchmark, or ``""``; main's #1904 check."""
    return context_copy_refusal(*_argv(base), *_argv(candidate))


def pinned_speculative_refusal(base: LaunchConfig, candidate: LaunchConfig) -> str:
    """Why ``candidate`` may not change the draft or simulated acceptance its base pins, or ``""``; main's #1885."""
    try:
        refuse_unpinned_speculative_change(*_argv(base), *_argv(candidate))
    except SpeculativeDecodingPinnedError as error:
        return str(error)
    return ""


def _server_log(round_dir: Path) -> str | None:
    """The server log of a round: the one in its directory, else the newest below it (a warm-up round's launch)."""
    direct = round_dir / "server.log"
    if direct.is_file():
        return str(direct)
    found = sorted(round_dir.rglob("server.log"), key=lambda path: path.stat().st_mtime)
    return str(found[-1]) if found else None


def baseline_prefix_cache_refusal(round_dir: Path) -> str:
    """Why an AgentX baseline round served with the prefix cache off may not end ``succeeded``, or ``""``.

    ``round_dir`` is the inference_suite run_dir the round landed in; main's #1884 check reads the launch its
    ``launch.json`` records. A round ``round_measurement`` does not call a valid measurement is not the succeeded round
    main checks, and is left alone. A run_dir whose report or launch breaks its contract, or a directory that is no
    run_dir, raises :class:`RoundUnreadable`.
    """
    if not is_round_folder(round_dir):
        raise RoundUnreadable(f"{round_dir}: not an inference_suite run_dir")
    return _suite_round_refusal(round_dir, _server_log(round_dir))


def _suite_round_refusal(run_dir: Path, log: str | None) -> str:
    try:
        folder = RoundFolder.read(run_dir)
    except (OSError, ValueError) as error:
        raise RoundUnreadable(f"{run_dir}: {error}") from error
    if folder.launch is None or not folder.measurement()["valid_measurement"]:
        return ""
    framework = folder.launch.framework.name.value
    return _cacheless_refusal(
        {
            "framework": framework,
            "requested_server_args": shlex.join(folder.launch.argv),
            "observed_server_config": observed_server_config_from_log(log, framework) if log else {},
        }
    )


def _cacheless_refusal(evidence: dict[str, Any]) -> str:
    measured: dict[str, Any] = {"status": "succeeded", "launch_evidence": evidence}
    refuse_cacheless_agentx_baseline(measured)
    return str(measured.get("error") or "") if measured["status"] != "succeeded" else ""


@dataclass(frozen=True)
class FakeGain:
    """One known fake gain: how it shows, what refuses it, and where."""

    name: str
    shows_as: str
    refusal_code: str  # the ``RequestRejected`` code, or the prefix of the failure reason the executor writes
    check: Callable[..., str]
    where: str  # "acceptance" (before anything launches) or "executor" (the round that ran)


FAKE_GAINS = (
    FakeGain(
        name="context_copy_draft_on_synthetic",
        shows_as="A speculative draft that copies the context accepts nearly every token of a synthetic benchmark.",
        refusal_code=CONTEXT_COPY_REFUSAL,
        check=launch_context_copy_refusal,
        where="acceptance",
    ),
    FakeGain(
        name=SPECULATIVE_DECODING_PINNED,
        shows_as="Changing a draft whose acceptance length the parent simulates moves the number without a real draft.",
        refusal_code=SPECULATIVE_DECODING_PINNED,
        check=pinned_speculative_refusal,
        where="acceptance",
    ),
    FakeGain(
        name="agentx_baseline_prefix_cache_off",
        shows_as="An AgentX baseline served with the prefix cache off makes every later candidate look faster.",
        refusal_code=PREFIX_CACHE_OFF_ERROR_CLASS,
        check=baseline_prefix_cache_refusal,
        where="executor",
    ),
)
