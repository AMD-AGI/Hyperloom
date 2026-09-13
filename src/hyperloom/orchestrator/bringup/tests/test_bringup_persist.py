# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The persisted observation survives the trip to disk, and a missing one is named."""

from __future__ import annotations

from hyperloom.common.bringup import LadderStage, failure_digest
from hyperloom.orchestrator.bringup import (
    DEGRADED_NO_PATH,
    DEGRADED_UNREADABLE,
    load_boot_observation,
    observe_bringup,
    verdict_of,
    write_boot_observation,
)

_IMPORT_FAILURE = (
    "Traceback (most recent call last):\n"
    '  File "/opt/vllm/vllm/model_executor/ops.py", line 41, in init\n'
    "ImportError: cannot import name '_C' from 'vllm'\n"
)


def _persist(tmp_path):
    session = tmp_path / "session"
    slot = tmp_path / "round"
    slot.mkdir(parents=True)
    verdict = observe_bringup(server_log=_IMPORT_FAILURE, server_elapsed_sec=12.5, session_dir=session)
    path = write_boot_observation(verdict.observation, session_dir=session, output_dir=slot, attempt=0)
    return verdict, path


def test_written_observation_reloads_identically(tmp_path) -> None:
    verdict, path = _persist(tmp_path)
    assert path

    loaded = load_boot_observation(path)
    assert loaded.degraded == ""
    assert loaded.observation == verdict.observation
    assert loaded.observation.stage_failed is LadderStage.IMPORT
    assert loaded.observation.server_elapsed_sec == 12.5
    assert failure_digest(loaded.observation) == failure_digest(verdict.observation)


def test_reloaded_observation_yields_the_same_signature(tmp_path) -> None:
    verdict, path = _persist(tmp_path)

    recovered = verdict_of(load_boot_observation(path).observation)
    assert recovered.signature.kind == verdict.signature.kind


def test_absent_path_and_missing_file_degrade_by_name(tmp_path) -> None:
    empty = load_boot_observation("")
    assert (empty.observation, empty.degraded) == (None, DEGRADED_NO_PATH)

    gone = load_boot_observation(tmp_path / "nothing.json")
    assert gone.observation is None
    assert gone.degraded == DEGRADED_UNREADABLE
    assert gone.path == str(tmp_path / "nothing.json")
